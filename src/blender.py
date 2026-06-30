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
    overlap_mask: Optional[np.ndarray] = None,
    confidence_scope: str = "overlap_only",
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
        confidence = confidence_mask.astype(np.float32) / 255.0
        if confidence_scope == "full_frame" or overlap_mask is None:
            weight *= confidence
        else:
            overlap_binary = overlap_mask > 0
            weight[overlap_binary] *= confidence[overlap_binary]
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
    confidence_scope = str(config.get("confidence_scope", "overlap_only")).lower()
    if confidence_scope not in {"overlap_only", "full_frame"}:
        raise ConfigError(f"Unsupported confidence_scope: {confidence_scope}")

    first_image = next(iter(warped_images.values()))
    accum = np.zeros(first_image.shape, dtype=np.float32)
    total_weight = np.zeros(first_image.shape[:2], dtype=np.float32)
    weight_maps: Dict[str, np.ndarray] = {}
    valid_binary = {
        camera_id: (valid_mask > 0).astype(np.uint8)
        for camera_id, valid_mask in valid_masks.items()
    }
    overlap_masks: Dict[str, np.ndarray] = {}
    if confidence_scope == "overlap_only":
        total_valid = np.zeros(first_image.shape[:2], dtype=np.uint16)
        for binary in valid_binary.values():
            total_valid += binary.astype(np.uint16)
        overlap_union = np.where(total_valid > 1, 255, 0).astype(np.uint8)
        for camera_id, binary in valid_binary.items():
            overlap_masks[camera_id] = np.where((overlap_union > 0) & (binary > 0), 255, 0).astype(np.uint8)
    else:
        overlap_masks = {camera_id: np.zeros(first_image.shape[:2], dtype=np.uint8) for camera_id in warped_images}

    for camera_id, image in warped_images.items():
        building_mask = building_masks.get(camera_id) if exclude_building else None
        weight = build_weight_map(
            valid_masks[camera_id],
            method,
            feather_radius,
            building_mask,
            confidence_masks.get(camera_id),
            overlap_masks.get(camera_id),
            confidence_scope,
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
    strategy: str = "reference_overlay",
) -> np.ndarray:
    if building_mask is None:
        return base_image
    normalized = str(strategy).lower()
    if normalized == "exclude":
        return base_image
    if normalized not in {"weak_blend", "reference_overlay"}:
        raise ConfigError(f"Unsupported building_strategy: {strategy}")
    output = base_image.copy().astype(np.float32)
    reference = reference_image.astype(np.float32)
    mask = (building_mask > 0)[..., None].astype(np.float32)
    overlay_alpha = alpha if normalized == "weak_blend" else 1.0
    blended = output * (1.0 - overlay_alpha) + reference * overlay_alpha
    output = output * (1.0 - mask) + blended * mask
    return np.clip(output, 0, 255).astype(np.uint8)
