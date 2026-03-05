from __future__ import annotations

from io import BytesIO
from typing import Iterable

import cv2
import numpy as np
from PIL import Image
from psd_tools import PSDImage
from psd_tools.api.layers import PixelLayer


class PSDService:
    """Export RGBA layers into a PSD using psd-tools."""

    def export(self, canvas_size: tuple[int, int], layers: Iterable[dict]) -> bytes:
        width, height = canvas_size
        psd = PSDImage.new(mode="RGBA", size=(width, height), color=(0, 0, 0, 0))

        for idx, layer in enumerate(layers):
            rgba = layer["rgba"]
            name = layer.get("name", f"layer_{idx}")
            if rgba.dtype != np.uint8:
                rgba = np.clip(rgba, 0, 255).astype(np.uint8)
            pil = Image.fromarray(cv2.cvtColor(rgba, cv2.COLOR_BGRA2RGBA), mode="RGBA")
            pixel_layer = PixelLayer.frompil(pil, psd, name=name, top=0, left=0)
            psd.append(pixel_layer)

        bio = BytesIO()
        psd.save(bio)
        bio.seek(0)
        return bio.getvalue()
