from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np


@dataclass
class WarpResult:
    warped_image: np.ndarray
    valid_mask: np.ndarray
    warped_building_mask: Optional[np.ndarray]
    warped_confidence_mask: Optional[np.ndarray]


def warp_to_canvas(
    image: np.ndarray,
    homography: np.ndarray,
    canvas_size: tuple[int, int],
    building_mask: Optional[np.ndarray] = None,
    confidence_mask: Optional[np.ndarray] = None,
) -> WarpResult:
    width, height = canvas_size
    warped_image = cv2.warpPerspective(image, homography, (width, height))

    source_valid = np.full(image.shape[:2], 255, dtype=np.uint8)
    valid_mask = cv2.warpPerspective(source_valid, homography, (width, height))
    valid_mask = np.where(valid_mask > 0, 255, 0).astype(np.uint8)

    warped_building_mask = None
    if building_mask is not None:
        warped_building_mask = cv2.warpPerspective(building_mask, homography, (width, height))
        warped_building_mask = np.where(warped_building_mask > 0, 255, 0).astype(np.uint8)

    warped_confidence_mask = None
    if confidence_mask is not None:
        warped_confidence_mask = cv2.warpPerspective(confidence_mask, homography, (width, height))
        warped_confidence_mask = np.where(warped_confidence_mask > 0, warped_confidence_mask, 0).astype(np.uint8)

    return WarpResult(
        warped_image=warped_image,
        valid_mask=valid_mask,
        warped_building_mask=warped_building_mask,
        warped_confidence_mask=warped_confidence_mask,
    )
