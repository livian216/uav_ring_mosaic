from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional

import cv2
import numpy as np


@dataclass
class WarpModel:
    model_type: str
    global_homography: np.ndarray
    metadata: Optional[dict[str, Any]] = None


@dataclass
class APAPModel:
    mesh_cols: int
    mesh_rows: int
    sigma: float
    min_weight: float
    source_points: np.ndarray
    target_points: np.ndarray
    local_homographies: np.ndarray
    target_shape: tuple[int, int]


def build_warp_model(
    model_type: str,
    global_homography: np.ndarray,
    config: Optional[Mapping[str, Any]] = None,
    source_points: Optional[np.ndarray] = None,
    target_points: Optional[np.ndarray] = None,
) -> WarpModel:
    normalized = (model_type or "global").lower()
    if normalized not in {"global", "apap"}:
        normalized = "global"
    metadata = {
        "requested_model": normalized,
        "source_points_count": 0 if source_points is None else int(len(source_points)),
        "target_points_count": 0 if target_points is None else int(len(target_points)),
        "config": dict(config or {}),
    }
    return WarpModel(model_type=normalized, global_homography=global_homography.astype(np.float32), metadata=metadata)


def _build_dlt_matrix(source_points: np.ndarray, target_points: np.ndarray) -> np.ndarray:
    rows = []
    for (x, y), (u, v) in zip(source_points, target_points):
        rows.append([-x, -y, -1.0, 0.0, 0.0, 0.0, u * x, u * y, u])
        rows.append([0.0, 0.0, 0.0, -x, -y, -1.0, v * x, v * y, v])
    return np.asarray(rows, dtype=np.float64)


def _solve_weighted_homography(
    source_points: np.ndarray,
    target_points: np.ndarray,
    center: np.ndarray,
    sigma: float,
    min_weight: float,
    fallback_h: np.ndarray,
) -> np.ndarray:
    distances = np.linalg.norm(target_points - center[None, :], axis=1)
    weights = np.exp(-(distances**2) / (sigma**2))
    weights = np.maximum(weights, min_weight)
    if np.count_nonzero(weights > min_weight * 1.01) < 4:
        return fallback_h

    a_matrix = _build_dlt_matrix(source_points, target_points)
    repeated = np.repeat(np.sqrt(weights), 2)
    weighted_a = a_matrix * repeated[:, None]
    _, _, vh = np.linalg.svd(weighted_a, full_matrices=False)
    h = vh[-1].reshape(3, 3)
    if abs(h[2, 2]) < 1e-8:
        return fallback_h
    h /= h[2, 2]
    if not np.all(np.isfinite(h)):
        return fallback_h
    return h.astype(np.float32)


def build_apap_model(
    source_points: np.ndarray,
    target_points: np.ndarray,
    target_shape: tuple[int, int],
    global_homography: np.ndarray,
    config: Optional[Mapping[str, Any]] = None,
) -> APAPModel:
    cfg = dict(config or {})
    mesh_cols = int(cfg.get("mesh_cols", 16))
    mesh_rows = int(cfg.get("mesh_rows", 16))
    sigma = float(cfg.get("sigma", 220.0))
    min_weight = float(cfg.get("min_weight", 0.001))
    target_height, target_width = target_shape[:2]

    x_edges = np.linspace(0.0, float(target_width), mesh_cols + 1, dtype=np.float32)
    y_edges = np.linspace(0.0, float(target_height), mesh_rows + 1, dtype=np.float32)
    local_homographies = np.zeros((mesh_rows, mesh_cols, 3, 3), dtype=np.float32)

    for row in range(mesh_rows):
        for col in range(mesh_cols):
            center = np.array(
                [(x_edges[col] + x_edges[col + 1]) * 0.5, (y_edges[row] + y_edges[row + 1]) * 0.5],
                dtype=np.float32,
            )
            local_homographies[row, col] = _solve_weighted_homography(
                source_points,
                target_points,
                center,
                sigma,
                min_weight,
                global_homography,
            )

    return APAPModel(
        mesh_cols=mesh_cols,
        mesh_rows=mesh_rows,
        sigma=sigma,
        min_weight=min_weight,
        source_points=source_points.astype(np.float32),
        target_points=target_points.astype(np.float32),
        local_homographies=local_homographies,
        target_shape=(target_height, target_width),
    )


def _remap_patch(
    src: np.ndarray,
    dst: np.ndarray,
    homography: np.ndarray,
    x0: int,
    x1: int,
    y0: int,
    y1: int,
    interpolation: int,
    border_value,
) -> None:
    if x1 <= x0 or y1 <= y0:
        return
    inv_h = np.linalg.inv(homography).astype(np.float32)
    grid_x, grid_y = np.meshgrid(np.arange(x0, x1, dtype=np.float32), np.arange(y0, y1, dtype=np.float32))
    ones = np.ones_like(grid_x)
    target = np.stack([grid_x, grid_y, ones], axis=-1).reshape(-1, 3).T
    source = inv_h @ target
    source /= np.maximum(source[2:3, :], 1e-8)
    map_x = source[0].reshape(y1 - y0, x1 - x0).astype(np.float32)
    map_y = source[1].reshape(y1 - y0, x1 - x0).astype(np.float32)
    patch = cv2.remap(
        src,
        map_x,
        map_y,
        interpolation=interpolation,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border_value,
    )
    dst[y0:y1, x0:x1] = patch


def warp_image_with_apap(
    image: np.ndarray,
    model: APAPModel,
    interpolation: int = cv2.INTER_LINEAR,
    border_value=0,
) -> np.ndarray:
    target_height, target_width = model.target_shape
    output_shape = (target_height, target_width) if image.ndim == 2 else (target_height, target_width, image.shape[2])
    warped = np.zeros(output_shape, dtype=image.dtype)
    x_edges = np.linspace(0, target_width, model.mesh_cols + 1, dtype=np.int32)
    y_edges = np.linspace(0, target_height, model.mesh_rows + 1, dtype=np.int32)

    for row in range(model.mesh_rows):
        for col in range(model.mesh_cols):
            _remap_patch(
                image,
                warped,
                model.local_homographies[row, col],
                int(x_edges[col]),
                int(x_edges[col + 1]),
                int(y_edges[row]),
                int(y_edges[row + 1]),
                interpolation,
                border_value,
            )
    return warped
