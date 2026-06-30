from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import cv2
import numpy as np

from apap import build_apap_model, warp_image_with_apap
from blender import blend_warped_images, overlay_building_region
from calibration import undistort_image
from homography import build_camera_graph, ensure_homography_bundle, save_homographies
from mosaic_canvas import warp_to_canvas
from utils import (
    ConfigError,
    get_input_paths,
    get_mask_paths,
    get_output_path,
    info,
    load_config,
    load_image,
    parse_args,
    resize_for_preview,
    save_image,
    validate_mask_shape,
    warning,
)


def load_optional_masks(config: dict, root: Path, image_paths: Dict[str, Path]) -> Dict[str, Optional[np.ndarray]]:
    use_masks = bool(config.get("masks", {}).get("use_building_mask", True))
    mask_paths = get_mask_paths(config, root)
    masks: Dict[str, Optional[np.ndarray]] = {}
    for camera_id, image_path in image_paths.items():
        if not use_masks or camera_id not in mask_paths or not mask_paths[camera_id].exists():
            warning(f"Building mask missing for {camera_id}; continuing without it.")
            masks[camera_id] = None
            continue
        image = load_image(image_path)
        mask = load_image(mask_paths[camera_id], cv2.IMREAD_GRAYSCALE)
        validate_mask_shape(mask, image.shape[:2])
        masks[camera_id] = mask
    return masks


def main() -> None:
    args = parse_args()
    config, root = load_config(args.config)
    image_paths = get_input_paths(config, root, mode="image")
    control_points_path = get_output_path(root, config["homography"]["control_points_file"])
    homography_path = get_output_path(root, config["homography"]["homography_file"])
    camera_cfg = config.get("camera", {})
    use_undistort = bool(camera_cfg.get("use_undistort", False))
    calibration_file = (root / camera_cfg.get("calibration_file", "configs/cameras.yaml")).resolve()

    masks = load_optional_masks(config, root, image_paths)
    loaded_images = {camera_id: load_image(path) for camera_id, path in image_paths.items()}
    image_shapes = {camera_id: image.shape[:2] for camera_id, image in loaded_images.items()}
    source_confidence_masks: Dict[str, Optional[np.ndarray]] = {camera_id: None for camera_id in image_paths}
    pair_models: Dict[str, dict[str, Any]] = {}
    canvas_metadata: Dict[str, Any] = {}
    if config["homography"].get("mode", "manual") == "auto":
        auto_result = build_camera_graph(
            loaded_images,
            masks=masks,
            config=config,
            image_shapes=image_shapes,
            debug_dir=root / "outputs" / "homographies" / "debug",
        )
        homographies = auto_result["homographies"]
        canvas_metadata = auto_result.get("canvas_metadata", {})
        source_confidence_masks.update(auto_result.get("confidence_masks", {}))
        pair_models = auto_result.get("pair_models", {})
        save_homographies(
            homography_path,
            homographies,
            {
                **canvas_metadata,
                "reference_camera": auto_result.get("reference_camera"),
                "parent_map": auto_result.get("parent_map", {}),
                "layout_graph": auto_result.get("layout_graph", {}),
            },
        )
        if auto_result.get("pair_reports"):
            from utils import save_yaml
            save_yaml(root / "outputs" / "homographies" / "debug" / "pair_quality.yaml", auto_result["pair_reports"])
    else:
        homographies, canvas_metadata = ensure_homography_bundle(
            control_points_path,
            homography_path,
            bool(config["homography"].get("compute_if_missing", True)),
            config=config,
            image_shapes=image_shapes,
            building_masks=masks,
        )

    width = int(canvas_metadata.get("canvas_width", config["canvas"]["width"]))
    height = int(canvas_metadata.get("canvas_height", config["canvas"]["height"]))
    canvas_size = (width, height)

    warped_images: Dict[str, np.ndarray] = {}
    valid_masks: Dict[str, np.ndarray] = {}
    warped_building_masks: Dict[str, Optional[np.ndarray]] = {}
    warped_confidence_masks: Dict[str, Optional[np.ndarray]] = {}
    camera_order = list(config["homography"].get("camera_order", image_paths.keys()))
    camera_to_target = {camera_order[i]: camera_order[i - 1] for i in range(1, len(camera_order))}
    warp_model_type = str(config["homography"].get("warp_model", "global")).lower()

    debug_dir = root / "outputs" / "mosaics" / "debug"
    for camera_id, image_path in image_paths.items():
        image = loaded_images[camera_id]
        image = undistort_image(image, camera_id, use_undistort, calibration_file)
        building_mask = masks[camera_id]
        confidence_mask = source_confidence_masks.get(camera_id)
        homography_for_canvas = homographies[camera_id]

        if warp_model_type == "apap" and camera_id in camera_to_target:
            target_camera = camera_to_target[camera_id]
            pair_key = f"{camera_id}__to__{target_camera}"
            pair_model = pair_models.get(pair_key)
            if pair_model:
                apap_model = build_apap_model(
                    np.asarray(pair_model["source_points"], dtype=np.float32),
                    np.asarray(pair_model["target_points"], dtype=np.float32),
                    loaded_images[target_camera].shape[:2],
                    np.asarray(pair_model["global_homography"], dtype=np.float32),
                    pair_model.get("apap_config", {}),
                )
                image = warp_image_with_apap(image, apap_model, interpolation=cv2.INTER_LINEAR, border_value=0)
                if building_mask is not None:
                    building_mask = warp_image_with_apap(building_mask, apap_model, interpolation=cv2.INTER_NEAREST, border_value=0)
                if confidence_mask is not None:
                    confidence_mask = warp_image_with_apap(confidence_mask, apap_model, interpolation=cv2.INTER_LINEAR, border_value=0)
                homography_for_canvas = homographies[target_camera]
                if config["debug"].get("save_warped_images", True):
                    save_image(debug_dir / f"{camera_id}_apap_target_plane.jpg", image)

        result = warp_to_canvas(
            image,
            homography_for_canvas,
            canvas_size,
            building_mask,
            confidence_mask,
        )
        warped_images[camera_id] = result.warped_image
        valid_masks[camera_id] = result.valid_mask
        warped_building_masks[camera_id] = result.warped_building_mask
        warped_confidence_masks[camera_id] = result.warped_confidence_mask

        if config["debug"].get("save_warped_images", True):
            save_image(debug_dir / f"{camera_id}_warped.jpg", result.warped_image)
        if config["debug"].get("save_masks", True):
            save_image(debug_dir / f"{camera_id}_valid_mask.png", result.valid_mask)
            if result.warped_building_mask is not None:
                save_image(debug_dir / f"{camera_id}_building_mask.png", result.warped_building_mask)
            if result.warped_confidence_mask is not None:
                save_image(debug_dir / f"{camera_id}_confidence_mask.png", result.warped_confidence_mask)

    blended, weight_maps, total_weight = blend_warped_images(
        warped_images,
        valid_masks,
        warped_building_masks,
        warped_confidence_masks,
        config["blending"],
    )

    reference_camera = config["blending"].get("building_reference_camera", next(iter(warped_images)))
    overlay_alpha = float(config["blending"].get("building_overlay_alpha", 0.45))
    building_strategy = str(config["blending"].get("building_strategy", "reference_overlay"))
    if reference_camera not in warped_images:
        raise ConfigError(f"Building reference camera not found: {reference_camera}")
    final_mosaic = overlay_building_region(
        blended,
        warped_images[reference_camera],
        warped_building_masks.get(reference_camera),
        overlay_alpha,
        building_strategy,
    )

    if config["debug"].get("save_weight_maps", True):
        for camera_id, weight in weight_maps.items():
            vis = np.clip(weight * 255.0, 0, 255).astype(np.uint8)
            save_image(debug_dir / f"{camera_id}_weight.png", vis)
        total_vis = np.clip((total_weight / max(total_weight.max(), 1e-6)) * 255.0, 0, 255).astype(np.uint8)
        save_image(debug_dir / "total_weight.png", total_vis)

    debug_summary = {
        "canvas_size": {"width": width, "height": height},
        "canvas_metadata": canvas_metadata,
        "coverage": {},
    }
    for camera_id, valid_mask in valid_masks.items():
        ys, xs = np.where(valid_mask > 0)
        if len(xs) == 0:
            continue
        debug_summary["coverage"][camera_id] = {
            "valid_bbox": [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())],
            "valid_area": int((valid_mask > 0).sum()),
            "weight_area": int((weight_maps[camera_id] > 1e-6).sum()),
        }
    from utils import save_yaml
    save_yaml(debug_dir / "coverage_summary.yaml", debug_summary)

    output_path = root / "outputs" / "mosaics" / "mosaic_result.jpg"
    preview_path = root / "outputs" / "mosaics" / "mosaic_preview.jpg"
    save_image(output_path, final_mosaic)
    save_image(preview_path, resize_for_preview(final_mosaic, float(config["canvas"].get("preview_scale", 1.0))))
    info(f"Saved mosaic to {output_path}")


if __name__ == "__main__":
    main()
