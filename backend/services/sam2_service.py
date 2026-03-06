from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np


@dataclass
class SAM2Service:
    """
    SAM2-compatible service.

    - If SAM2 is available, you can plug inference into `generate_candidates`.
    - In this repository we provide a robust OpenCV fallback that creates
      multiple object proposals suitable for click-select workflows.
    """

    click_radius: int = 6
    max_candidates: int = 48
    min_area_ratio: float = 0.0005
    _sam2_available: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        # Keep optional import local to avoid hard dependency for environments
        # that only need fallback behavior.
        try:
            import sam2  # type: ignore # noqa: F401

            self._sam2_available = True
        except Exception:
            self._sam2_available = False

    def generate_candidates(self, image_bgr: np.ndarray) -> list[np.ndarray]:
        if image_bgr is None or image_bgr.size == 0:
            raise ValueError("Empty image provided")

        # Placeholder for SAM2 integration path.
        if self._sam2_available:
            # TODO: Replace with real SAM2 automatic mask generation.
            # Fallback is intentionally used until model wiring is configured.
            pass

        return self._fallback_candidates(image_bgr)

    def select_candidate_by_click(self, candidates: list[np.ndarray], x: int, y: int) -> np.ndarray:
        if not candidates:
            raise ValueError("No candidates available")

        h, w = candidates[0].shape[:2]
        x = int(np.clip(x, 0, w - 1))
        y = int(np.clip(y, 0, h - 1))

        # Prefer smallest mask containing clicked point so detailed parts
        # (eye, mouth, accessories) are easier to pick.
        selected: np.ndarray | None = None
        selected_area = None
        for mask in candidates:
            if mask[y, x] != 1:
                continue
            area = int(mask.sum())
            if selected is None or area < selected_area:
                selected = mask
                selected_area = area

        if selected is not None:
            return selected.astype(np.uint8)

        # Fallback: nearest region center from all candidates.
        click_xy = np.array([x, y], dtype=np.float32)
        best_idx = 0
        best_dist = float("inf")
        for idx, mask in enumerate(candidates):
            ys, xs = np.where(mask == 1)
            if len(xs) == 0:
                continue
            center = np.array([xs.mean(), ys.mean()], dtype=np.float32)
            dist = float(np.linalg.norm(center - click_xy))
            if dist < best_dist:
                best_dist = dist
                best_idx = idx

        return candidates[best_idx].astype(np.uint8)

    def _fallback_candidates(self, image_bgr: np.ndarray) -> list[np.ndarray]:
        h, w = image_bgr.shape[:2]
        min_area = max(64, int(h * w * self.min_area_ratio))

        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)

        candidates: list[np.ndarray] = []

        # 1) Superpixel-like regions from watershed markers.
        grad_x = cv2.Sobel(blur, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(blur, cv2.CV_32F, 0, 1, ksize=3)
        grad_mag = cv2.magnitude(grad_x, grad_y)
        grad_u8 = cv2.normalize(grad_mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        _, sure_fg = cv2.threshold(255 - grad_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        sure_fg = cv2.morphologyEx(sure_fg, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=2)

        num_markers, markers = cv2.connectedComponents(sure_fg)
        if num_markers > 1:
            markers = markers + 1
            markers[sure_fg == 0] = 0
            ws_markers = cv2.watershed(image_bgr.copy(), markers.astype(np.int32))
            unique_labels = [v for v in np.unique(ws_markers) if v > 1]
            for label in unique_labels:
                mask = (ws_markers == label).astype(np.uint8)
                area = int(mask.sum())
                if area >= min_area:
                    candidates.append(mask)

        # 2) Multi-threshold connected components to catch missing parts.
        for mode in [cv2.THRESH_BINARY, cv2.THRESH_BINARY_INV]:
            _, th = cv2.threshold(blur, 0, 255, mode + cv2.THRESH_OTSU)
            th = cv2.morphologyEx(th, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
            count, labels, stats, _ = cv2.connectedComponentsWithStats(th, connectivity=8)
            for i in range(1, count):
                area = int(stats[i, cv2.CC_STAT_AREA])
                if area < min_area:
                    continue
                mask = (labels == i).astype(np.uint8)
                candidates.append(mask)

        # 3) Deduplicate near-identical masks via IoU.
        deduped: list[np.ndarray] = []
        for mask in sorted(candidates, key=lambda m: int(m.sum())):
            is_duplicate = False
            for existing in deduped:
                inter = int(np.logical_and(mask, existing).sum())
                union = int(np.logical_or(mask, existing).sum())
                if union == 0:
                    continue
                if inter / union > 0.9:
                    is_duplicate = True
                    break
            if not is_duplicate:
                deduped.append(mask)
            if len(deduped) >= self.max_candidates:
                break

        if not deduped:
            # Ensure at least one candidate exists.
            fallback = np.ones((h, w), dtype=np.uint8)
            deduped = [fallback]

        return deduped
