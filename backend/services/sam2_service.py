from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class SAM2Service:
    """
    Lightweight click-based segmentation service.

    If real SAM2 dependencies are available in your environment, you can extend this
    class to load the model and replace `_fallback_segment` with SAM2 inference.
    """

    click_radius: int = 6

    def segment_by_click(self, image_bgr: np.ndarray, x: int, y: int, positive: bool = True) -> np.ndarray:
        if image_bgr is None or image_bgr.size == 0:
            raise ValueError("Empty image provided")

        h, w = image_bgr.shape[:2]
        x = int(np.clip(x, 0, w - 1))
        y = int(np.clip(y, 0, h - 1))

        mask = self._fallback_segment(image_bgr, x, y)
        if not positive:
            mask = 1 - mask
        return mask.astype(np.uint8)

    def _fallback_segment(self, image_bgr: np.ndarray, x: int, y: int) -> np.ndarray:
        """GrabCut-based fallback that behaves like click segmentation."""
        h, w = image_bgr.shape[:2]
        gc_mask = np.full((h, w), cv2.GC_PR_BGD, dtype=np.uint8)

        x0 = max(0, x - self.click_radius)
        y0 = max(0, y - self.click_radius)
        x1 = min(w, x + self.click_radius + 1)
        y1 = min(h, y + self.click_radius + 1)
        gc_mask[y0:y1, x0:x1] = cv2.GC_FGD

        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)

        cv2.grabCut(image_bgr, gc_mask, None, bgd_model, fgd_model, 4, cv2.GC_INIT_WITH_MASK)

        binary = np.where(
            (gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD),
            1,
            0,
        ).astype(np.uint8)

        kernel = np.ones((3, 3), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)
        return binary
