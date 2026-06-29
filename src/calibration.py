from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import cv2
import numpy as np

from utils import load_yaml


def load_calibration_data(path: Path) -> Mapping[str, Any]:
    if not path.exists():
        return {}
    data = load_yaml(path)
    return data.get("cameras", {}) if isinstance(data, dict) else {}


def undistort_image(
    image: np.ndarray,
    camera_id: str,
    use_undistort: bool,
    calibration_file: Path,
) -> np.ndarray:
    if not use_undistort:
        return image
    calibration_data = load_calibration_data(calibration_file)
    params = calibration_data.get(camera_id)
    if not params:
        return image
    camera_matrix = np.array(params.get("camera_matrix", []), dtype=np.float32)
    dist_coeffs = np.array(params.get("dist_coeffs", []), dtype=np.float32)
    if camera_matrix.size != 9:
        return image
    return cv2.undistort(image, camera_matrix.reshape(3, 3), dist_coeffs)
