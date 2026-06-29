from __future__ import annotations

from typing import Dict, Mapping, Optional

import cv2
import numpy as np

from utils import ConfigError


def _soft_weight_from_mask(valid_mask: np.ndarray, feather_radius: int) -> np.ndarray:
    binary = (valid_mask > 0).astype(np.uint8)
    if feather_radius <= 0:
        return binary.astype(np.float32)
    distance = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    weight = np.clip(distance / float(max(feather_radius, 1)), 0.0, 1.0)
    weight[binary == 0] = 0.0
    return weight.astype(np.float32)


def build_weight_map(
    valid_mask: np.ndarray,
    method: str,
    feather_radius: int,
    exclude_mask: Optional[np.ndarray] = None,
    confidence_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    if method not in {"average", "alpha", "feather"}:
        raise ConfigError(f"Unsupported blending method: {method}")
    if method == "feather":
        weight = _soft_weight_from_mask(valid_mask, feather_radius)
    else:
        weight = (valid_mask > 0).astype(np.float32)
    if exclude_mask is not None:
        weight[exclude_mask > 0] = 0.0
    if confidence_mask is not None:
        weight *= confidence_mask.astype(np.float32) / 255.0
    return weight


def blend_warped_images(
    warped_images: Mapping[str, np.ndarray],
    valid_masks: Mapping[str, np.ndarray],
    building_masks: Mapping[str, Optional[np.ndarray]],
    confidence_masks: Mapping[str, Optional[np.ndarray]],
    config: Mapping[str, object],
) -> tuple[np.ndarray, Dict[str, np.ndarray], np.ndarray]:
    method = str(config.get("method", "feather"))
    feather_radius = int(config.get("feather_radius", 25))
    exclude_building = bool(config.get("exclude_building_area", True))

    first_image = next(iter(warped_images.values()))
    accum = np.zeros(first_image.shape, dtype=np.float32)
    total_weight = np.zeros(first_image.shape[:2], dtype=np.float32)
    weight_maps: Dict[str, np.ndarray] = {}

    for camera_id, image in warped_images.items():
        building_mask = building_masks.get(camera_id) if exclude_building else None
        weight = build_weight_map(
            valid_masks[camera_id],
            method,
            feather_radius,
            building_mask,
            confidence_masks.get(camera_id),
        )
        weight_maps[camera_id] = weight
        accum += image.astype(np.float32) * weight[..., None]
        total_weight += weight

    safe_weight = np.where(total_weight > 1e-6, total_weight, 1.0)
    blended = accum / safe_weight[..., None]
    blended[total_weight <= 1e-6] = 0
    return blended.astype(np.uint8), weight_maps, total_weight


def overlay_building_region(
    base_image: np.ndarray,
    reference_image: np.ndarray,
    building_mask: Optional[np.ndarray],
    alpha: float,
) -> np.ndarray:
    if building_mask is None:
        return base_image
    output = base_image.copy().astype(np.float32)
    reference = reference_image.astype(np.float32)
    mask = (building_mask > 0)[..., None].astype(np.float32)
    blended = output * (1.0 - alpha) + reference * alpha
    output = output * (1.0 - mask) + blended * mask
    return np.clip(output, 0, 255).astype(np.uint8)
