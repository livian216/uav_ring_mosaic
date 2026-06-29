from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Tuple

import cv2
import numpy as np
import yaml


class ConfigError(RuntimeError):
    """Raised when configuration or input data is invalid."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to YAML config file.")
    return parser.parse_args()


def load_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise ConfigError(f"YAML file not found: {path}")
    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


def save_yaml(path: Path, data: Mapping[str, Any]) -> None:
    ensure_parent(path)
    with path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(dict(data), file, allow_unicode=True, sort_keys=False)


def resolve_path(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (root / path).resolve()


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def load_config(config_path: str | Path) -> tuple[Dict[str, Any], Path]:
    config_file = Path(config_path).resolve()
    config = load_yaml(config_file)
    root = config_file.parent.parent.resolve()
    prepare_output_dirs(root)
    return config, root


def prepare_output_dirs(root: Path) -> None:
    for rel in [
        "outputs",
        "outputs/homographies",
        "outputs/mosaics",
        "outputs/mosaics/debug",
        "outputs/videos",
        "data/masks",
        "data/videos",
    ]:
        ensure_dir(root / rel)


def get_camera_ids(config: Mapping[str, Any], mode: str | None = None) -> List[str]:
    input_cfg = config.get("input", {})
    selected_mode = mode or input_cfg.get("mode", "image")
    key = "image_paths" if selected_mode == "image" else "video_paths"
    paths = input_cfg.get(key, {})
    if not isinstance(paths, dict) or not paths:
        raise ConfigError(f"No camera paths configured under input.{key}.")
    return list(paths.keys())


def get_input_paths(config: Mapping[str, Any], root: Path, mode: str | None = None) -> Dict[str, Path]:
    input_cfg = config.get("input", {})
    selected_mode = mode or input_cfg.get("mode", "image")
    key = "image_paths" if selected_mode == "image" else "video_paths"
    paths = input_cfg.get(key, {})
    if not isinstance(paths, dict) or not paths:
        raise ConfigError(f"No input paths configured under input.{key}.")
    return {camera_id: resolve_path(root, path) for camera_id, path in paths.items()}


def get_mask_paths(config: Mapping[str, Any], root: Path) -> Dict[str, Path]:
    mask_cfg = config.get("masks", {}).get("building_masks", {})
    return {camera_id: resolve_path(root, path) for camera_id, path in mask_cfg.items()}


def get_output_path(root: Path, relative_path: str) -> Path:
    path = resolve_path(root, relative_path)
    ensure_parent(path)
    return path


def load_image(path: Path, flag: int = cv2.IMREAD_COLOR) -> np.ndarray:
    image = cv2.imread(str(path), flag)
    if image is None:
        raise ConfigError(f"Failed to read image: {path}")
    return image


def save_image(path: Path, image: np.ndarray) -> None:
    ensure_parent(path)
    if not cv2.imwrite(str(path), image):
        raise ConfigError(f"Failed to save image: {path}")


def resize_for_preview(image: np.ndarray, scale: float) -> np.ndarray:
    if scale == 1.0:
        return image
    return cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)


def compute_display_scale(image_shape: Tuple[int, int], max_width: int, max_height: int) -> float:
    height, width = image_shape[:2]
    scale = min(max_width / float(width), max_height / float(height), 1.0)
    return max(scale, 1e-6)


def make_canvas(width: int, height: int, background_color: Iterable[int]) -> np.ndarray:
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    canvas[:] = tuple(background_color)
    return canvas


def validate_mask_shape(mask: np.ndarray, image_shape: Tuple[int, int]) -> None:
    if mask.shape[:2] != image_shape:
        raise ConfigError(
            f"Mask size mismatch. Expected {image_shape}, got {mask.shape[:2]}."
        )


def warning(message: str) -> None:
    print(f"[WARN] {message}")


def info(message: str) -> None:
    print(f"[INFO] {message}")
