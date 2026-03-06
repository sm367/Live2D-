from __future__ import annotations

import base64
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from backend.services import InpaintService, PSDService, SAM2Service


@dataclass
class LayerData:
    name: str
    mask: np.ndarray
    rgba: np.ndarray


@dataclass
class SessionData:
    image_bgr: np.ndarray
    inpainted_bgr: np.ndarray
    candidates: list[np.ndarray] = field(default_factory=list)
    layers: list[LayerData] = field(default_factory=list)


class ClickPayload(BaseModel):
    x: int = Field(..., ge=0)
    y: int = Field(..., ge=0)
    positive: bool = True
    layer_name: str | None = None


class InpaintPayload(BaseModel):
    expand_px: int = Field(16, ge=1, le=128)
    radius: int = Field(5, ge=1, le=25)


app = FastAPI(title="PNG to Live2D PSD Tool")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

sam2_service = SAM2Service()
inpaint_service = InpaintService()
psd_service = PSDService()
sessions: dict[str, SessionData] = {}

frontend_dir = Path(__file__).resolve().parents[1] / "frontend"
app.mount("/frontend", StaticFiles(directory=str(frontend_dir)), name="frontend")


def _mask_to_png_data_url(mask: np.ndarray, color: tuple[int, int, int] = (57, 169, 255), alpha: int = 110) -> str:
    h, w = mask.shape[:2]
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[mask == 1, 0] = color[2]
    rgba[mask == 1, 1] = color[1]
    rgba[mask == 1, 2] = color[0]
    rgba[mask == 1, 3] = alpha
    _, png = cv2.imencode(".png", rgba)
    return f"data:image/png;base64,{base64.b64encode(png.tobytes()).decode('utf-8')}"


@app.get("/")
def index() -> FileResponse:
    return FileResponse(frontend_dir / "index.html")


@app.post("/api/session")
async def create_session(file: UploadFile = File(...)) -> JSONResponse:
    if not file.filename.lower().endswith(".png"):
        raise HTTPException(status_code=400, detail="PNGファイルのみアップロード可能です")

    content = await file.read()
    image_np = np.frombuffer(content, np.uint8)
    image = cv2.imdecode(image_np, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise HTTPException(status_code=400, detail="画像の読み込みに失敗しました")

    if image.ndim == 2:
        image_bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.shape[2] == 4:
        image_bgr = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    else:
        image_bgr = image

    sid = str(uuid.uuid4())
    candidates = sam2_service.generate_candidates(image_bgr)
    sessions[sid] = SessionData(image_bgr=image_bgr, inpainted_bgr=image_bgr.copy(), candidates=candidates)

    h, w = image_bgr.shape[:2]
    candidate_previews = [_mask_to_png_data_url(mask) for mask in candidates[:24]]
    return JSONResponse(
        {
            "session_id": sid,
            "width": w,
            "height": h,
            "candidate_count": len(candidates),
            "candidate_previews": candidate_previews,
        }
    )


@app.post("/api/session/{session_id}/mask")
def add_mask(session_id: str, payload: ClickPayload) -> JSONResponse:
    session = sessions.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="セッションが見つかりません")

    if not session.candidates:
        session.candidates = sam2_service.generate_candidates(session.image_bgr)

    mask = sam2_service.select_candidate_by_click(session.candidates, payload.x, payload.y)
    if not payload.positive:
        mask = 1 - mask

    rgba = inpaint_service.cutout_rgba(session.image_bgr, mask)
    layer_name = payload.layer_name or f"part_{len(session.layers)+1}"
    session.layers.append(LayerData(name=layer_name, mask=mask, rgba=rgba))

    _, png = cv2.imencode(".png", rgba)
    b64 = base64.b64encode(png.tobytes()).decode("utf-8")
    return JSONResponse(
        {
            "layer_index": len(session.layers) - 1,
            "preview": f"data:image/png;base64,{b64}",
            "selected_mask_overlay": _mask_to_png_data_url(mask, color=(88, 239, 123), alpha=100),
        }
    )


@app.get("/api/session/{session_id}/candidates")
def list_candidates(session_id: str) -> JSONResponse:
    session = sessions.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="セッションが見つかりません")

    previews = [_mask_to_png_data_url(mask) for mask in session.candidates[:48]]
    return JSONResponse({"candidate_count": len(session.candidates), "candidate_previews": previews})


@app.post("/api/session/{session_id}/inpaint")
def run_inpaint(session_id: str, payload: InpaintPayload) -> JSONResponse:
    session = sessions.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="セッションが見つかりません")

    if not session.layers:
        raise HTTPException(status_code=400, detail="先にマスクを作成してください")

    merged_mask = np.zeros(session.image_bgr.shape[:2], dtype=np.uint8)
    for layer in session.layers:
        merged_mask = np.maximum(merged_mask, layer.mask.astype(np.uint8))

    expanded = inpaint_service.expand_mask(merged_mask, payload.expand_px)
    session.inpainted_bgr = inpaint_service.inpaint_margin(session.image_bgr, expanded, payload.radius)

    _, png = cv2.imencode(".png", session.inpainted_bgr)
    b64 = base64.b64encode(png.tobytes()).decode("utf-8")
    return JSONResponse({"preview": f"data:image/png;base64,{b64}"})


@app.post("/api/session/{session_id}/export")
def export_psd(session_id: str) -> Response:
    session = sessions.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="セッションが見つかりません")

    base_rgba = cv2.cvtColor(session.inpainted_bgr, cv2.COLOR_BGR2BGRA)
    base_rgba[:, :, 3] = 255
    layers = [{"name": "base_inpainted", "rgba": base_rgba}]
    layers += [{"name": layer.name, "rgba": layer.rgba} for layer in session.layers]

    h, w = session.image_bgr.shape[:2]
    psd_bytes = psd_service.export((w, h), layers)
    return Response(
        content=psd_bytes,
        media_type="image/vnd.adobe.photoshop",
        headers={"Content-Disposition": f'attachment; filename="live2d_{session_id}.psd"'},
    )


@app.delete("/api/session/{session_id}")
def delete_session(session_id: str) -> JSONResponse:
    sessions.pop(session_id, None)
    return JSONResponse({"ok": True})


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
