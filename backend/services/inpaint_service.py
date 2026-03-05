from __future__ import annotations

import cv2
import numpy as np


class InpaintService:
    """Edge-extension (のりしろ補完) and inpainting utilities."""

    def expand_mask(self, mask: np.ndarray, expand_px: int = 16) -> np.ndarray:
        expand_px = max(1, int(expand_px))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (expand_px, expand_px))
        return cv2.dilate(mask.astype(np.uint8), kernel, iterations=1)

    def inpaint_margin(self, image_bgr: np.ndarray, mask: np.ndarray, radius: int = 5) -> np.ndarray:
        """
        Fill selected region by OpenCV inpaint so separated layers have hidden margins.
        """
        mask_u8 = (mask > 0).astype(np.uint8) * 255
        return cv2.inpaint(image_bgr, mask_u8, float(radius), cv2.INPAINT_TELEA)

    def cutout_rgba(self, image_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
        alpha = (mask > 0).astype(np.uint8) * 255
        rgba = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2BGRA)
        rgba[:, :, 3] = alpha
        return rgba
