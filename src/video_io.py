from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Mapping, Optional

import cv2
import numpy as np

from utils import ConfigError


@dataclass
class VideoStreamState:
    capture: cv2.VideoCapture
    last_frame: Optional[np.ndarray] = None
    ended: bool = False


def open_video_streams(video_paths: Mapping[str, Path]) -> Dict[str, VideoStreamState]:
    streams: Dict[str, VideoStreamState] = {}
    for camera_id, path in video_paths.items():
        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            raise ConfigError(f"Failed to open video: {path}")
        streams[camera_id] = VideoStreamState(capture=capture)
    return streams


def read_stream_frame(state: VideoStreamState) -> tuple[bool, Optional[np.ndarray]]:
    if state.ended:
        return False, state.last_frame
    success, frame = state.capture.read()
    if success:
        state.last_frame = frame
        return True, frame
    state.ended = True
    return False, state.last_frame


def release_streams(streams: Mapping[str, VideoStreamState]) -> None:
    for state in streams.values():
        state.capture.release()
