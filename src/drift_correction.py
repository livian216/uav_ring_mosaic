from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


def estimate_translation_offset(
    current_frame: np.ndarray,
    reference_frame: np.ndarray,
    mask: Optional[np.ndarray] = None,
) -> Tuple[float, float]:
    """
    Estimate a small translation offset relative to a reference frame.
    V1 keeps this disabled and returns zero offset.
    """
    _ = current_frame, reference_frame, mask
    return 0.0, 0.0
