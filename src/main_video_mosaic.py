from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import cv2
import numpy as np

from blender import blend_warped_images, overlay_building_region
from calibration import undistort_image
from drift_correction import estimate_translation_offset
from homography import load_homographies
from mosaic_canvas import warp_to_canvas
from video_io import open_video_streams, read_stream_frame, release_streams
from utils import (
    ConfigError,
    get_input_paths,
    get_mask_paths,
    get_output_path,
    info,
    load_config,
    load_image,
    parse_args,
    validate_mask_shape,
    warning,
)


def load_optional_masks(config: dict, root: Path, frame_shapes: Dict[str, tuple[int, int]]) -> Dict[str, Optional[np.ndarray]]:
    use_masks = bool(config.get("masks", {}).get("use_building_mask", True))
    mask_paths = get_mask_paths(config, root)
    masks: Dict[str, Optional[np.ndarray]] = {}
    for camera_id, shape in frame_shapes.items():
        path = mask_paths.get(camera_id)
        if not use_masks or path is None or not path.exists():
            warning(f"Building mask missing for {camera_id}; continuing without it.")
            masks[camera_id] = None
            continue
        mask = load_image(path, cv2.IMREAD_GRAYSCALE)
        validate_mask_shape(mask, shape)
        masks[camera_id] = mask
    return masks


def main() -> None:
    args = parse_args()
    config, root = load_config(args.config)
    video_paths = get_input_paths(config, root, mode="video")
    homography_path = get_output_path(root, config["homography"]["homography_file"])
    if not homography_path.exists():
        raise ConfigError("Video mode requires a precomputed homography file.")
    homographies = load_homographies(homography_path)

    camera_cfg = config.get("camera", {})
    use_undistort = bool(camera_cfg.get("use_undistort", False))
    calibration_file = (root / camera_cfg.get("calibration_file", "configs/cameras.yaml")).resolve()

    streams = open_video_streams(video_paths)
    writer = None
    frozen_notified = set()
    reference_frames: Dict[str, np.ndarray] = {}

    try:
        frame_shapes: Dict[str, tuple[int, int]] = {}
        for camera_id, state in streams.items():
            success, frame = read_stream_frame(state)
            if not success or frame is None:
                raise ConfigError(f"Failed to read initial frame from {camera_id}.")
            reference_frames[camera_id] = frame.copy()
            frame_shapes[camera_id] = frame.shape[:2]

        masks = load_optional_masks(config, root, frame_shapes)
        width = int(config["canvas"]["width"])
        height = int(config["canvas"]["height"])
        canvas_size = (width, height)

        if bool(config["video"].get("save_output", True)):
            output_path = root / config["video"].get("output_path", "outputs/videos/mosaic_output.mp4")
            output_path.parent.mkdir(parents=True, exist_ok=True)
            writer = cv2.VideoWriter(
                str(output_path),
                cv2.VideoWriter_fourcc(*"mp4v"),
                float(config["video"].get("output_fps", 20)),
                (width, height),
            )

        while True:
            warped_images: Dict[str, np.ndarray] = {}
            valid_masks: Dict[str, np.ndarray] = {}
            warped_building_masks: Dict[str, Optional[np.ndarray]] = {}

            for camera_id, state in streams.items():
                success, frame = read_stream_frame(state)
                if not success:
                    if frame is None:
                        raise ConfigError(f"Video stream ended before any frame was cached: {camera_id}")
                    if camera_id not in frozen_notified:
                        warning(f"Video stream frozen on last frame: {camera_id}")
                        frozen_notified.add(camera_id)
                frame = undistort_image(frame, camera_id, use_undistort, calibration_file)
                estimate_translation_offset(frame, reference_frames[camera_id], masks.get(camera_id))
                result = warp_to_canvas(frame, homographies[camera_id], canvas_size, masks.get(camera_id))
                warped_images[camera_id] = result.warped_image
                valid_masks[camera_id] = result.valid_mask
                warped_building_masks[camera_id] = result.warped_building_mask

            blended, _, _ = blend_warped_images(
                warped_images,
                valid_masks,
                warped_building_masks,
                config["blending"],
            )

            reference_camera = config["blending"].get("building_reference_camera", next(iter(warped_images)))
            final_frame = overlay_building_region(
                blended,
                warped_images[reference_camera],
                warped_building_masks.get(reference_camera),
                float(config["blending"].get("building_overlay_alpha", 0.45)),
            )

            if writer is not None:
                writer.write(final_frame)

            if bool(config["video"].get("display_window", True)):
                cv2.imshow("Video Mosaic", final_frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            else:
                all_ended = all(state.ended for state in streams.values())
                if all_ended:
                    break

            if all(state.ended for state in streams.values()):
                break
    finally:
        if writer is not None:
            writer.release()
        release_streams(streams)
        cv2.destroyAllWindows()
        info("Video mosaic finished.")


if __name__ == "__main__":
    main()
