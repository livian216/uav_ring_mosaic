from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

import cv2
import numpy as np

from apap import build_apap_model, build_warp_model
from utils import ConfigError, ensure_dir, load_yaml, save_image, save_yaml, warning


def load_control_points(path: Path) -> Dict[str, Dict[str, Any]]:
    data = load_yaml(path)
    if not isinstance(data, dict) or not data:
        raise ConfigError(f"Control points file is empty: {path}")
    return data


def compute_homography(src_points: list[list[float]], dst_points: list[list[float]]) -> np.ndarray:
    if len(src_points) < 4 or len(dst_points) < 4:
        raise ConfigError("At least 4 source and destination points are required.")
    if len(src_points) != len(dst_points):
        raise ConfigError("Source and destination point counts must match.")
    src = np.asarray(src_points, dtype=np.float32)
    dst = np.asarray(dst_points, dtype=np.float32)
    if _point_spread_area(src) < 2000 or _point_spread_area(dst) < 2000:
        raise ConfigError("Control points are too concentrated or nearly collinear; spread points around the overlap region.")
    matrix, _ = cv2.findHomography(src, dst, cv2.RANSAC)
    if matrix is None:
        raise ConfigError("Homography estimation failed.")
    return matrix


def compute_homographies_from_points(control_points: Mapping[str, Any]) -> Dict[str, np.ndarray]:
    homographies: Dict[str, np.ndarray] = {}
    for camera_id, points in control_points.items():
        src_points = points.get("src_points", [])
        dst_points = points.get("dst_points", [])
        homographies[camera_id] = compute_homography(src_points, dst_points)
    return homographies


def _create_feature_detector(method: str, nfeatures: int):
    normalized = method.lower()
    if normalized == "sift" and hasattr(cv2, "SIFT_create"):
        return cv2.SIFT_create(nfeatures=nfeatures), cv2.NORM_L2, "sift"
    if normalized == "orb":
        return cv2.ORB_create(nfeatures=nfeatures), cv2.NORM_HAMMING, "orb"
    if hasattr(cv2, "SIFT_create"):
        warning("Requested feature method is unavailable; falling back to SIFT.")
        return cv2.SIFT_create(nfeatures=nfeatures), cv2.NORM_L2, "sift"
    warning("SIFT is unavailable; falling back to ORB.")
    return cv2.ORB_create(nfeatures=nfeatures), cv2.NORM_HAMMING, "orb"


def detect_features(image, mask=None, method="sift", nfeatures=4000):
    """Detect image features for automatic pairwise matching."""
    detector, _, resolved_method = _create_feature_detector(method, nfeatures)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    keypoints, descriptors = detector.detectAndCompute(gray, mask)
    return keypoints or [], descriptors, resolved_method


def match_features(image_a, image_b, mask_a=None, mask_b=None, method="sift", nfeatures=4000, ratio_test=0.75):
    """Match features between two overlapping images."""
    keypoints_a, descriptors_a, resolved_method = detect_features(image_a, mask_a, method, nfeatures)
    keypoints_b, descriptors_b, _ = detect_features(image_b, mask_b, resolved_method, nfeatures)
    if descriptors_a is None or descriptors_b is None:
        raise ConfigError("Failed to compute descriptors for automatic matching.")
    _, norm_type, _ = _create_feature_detector(resolved_method, nfeatures)
    matcher = cv2.BFMatcher(normType=norm_type)
    raw_matches = matcher.knnMatch(descriptors_a, descriptors_b, k=2)
    good_matches = []
    for pair in raw_matches:
        if len(pair) < 2:
            continue
        first, second = pair
        if first.distance < ratio_test * second.distance:
            good_matches.append(first)
    return {
        "method": resolved_method,
        "keypoints_a": keypoints_a,
        "keypoints_b": keypoints_b,
        "matches": good_matches,
    }


def estimate_pairwise_homography(
    keypoints_a,
    keypoints_b,
    matches,
    ransac_reproj_threshold=4.0,
    min_matches=40,
    min_inliers=25,
):
    """Estimate pairwise source->target homography from matched features."""
    if len(matches) < min_matches:
        raise ConfigError(f"Not enough feature matches: {len(matches)} < {min_matches}")
    src = np.float32([keypoints_a[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst = np.float32([keypoints_b[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    if _point_spread_area(src.reshape(-1, 2)) < 2000 or _point_spread_area(dst.reshape(-1, 2)) < 2000:
        raise ConfigError("Automatic matches are too concentrated or nearly collinear.")
    matrix, inlier_mask = cv2.findHomography(src, dst, cv2.RANSAC, ransac_reproj_threshold)
    if matrix is None or inlier_mask is None:
        raise ConfigError("Automatic homography estimation failed.")
    inliers = int(inlier_mask.sum())
    if inliers < min_inliers:
        raise ConfigError(f"Not enough inliers after RANSAC: {inliers} < {min_inliers}")
    return matrix, inlier_mask.ravel().astype(bool)


def _draw_match_visualization(image_a, image_b, keypoints_a, keypoints_b, matches, inlier_mask):
    selected = [match for match, keep in zip(matches, inlier_mask) if keep]
    selected = selected[:80]
    return cv2.drawMatches(
        image_a,
        keypoints_a,
        image_b,
        keypoints_b,
        selected,
        None,
        matchColor=(0, 255, 0),
        singlePointColor=(255, 0, 0),
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )


def _make_support_mask(image_shape, points: np.ndarray, dilate_size: int, blur_size: int) -> np.ndarray:
    height, width = image_shape[:2]
    mask = np.zeros((height, width), dtype=np.uint8)
    if len(points) < 3:
        return mask
    hull = cv2.convexHull(points.reshape(-1, 1, 2).astype(np.float32))
    if cv2.contourArea(hull) <= 1.0:
        return mask
    cv2.fillConvexPoly(mask, hull.astype(np.int32), 255)
    if dilate_size > 1:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_size, dilate_size))
        mask = cv2.dilate(mask, kernel)
    if blur_size > 1:
        if blur_size % 2 == 0:
            blur_size += 1
        mask = cv2.GaussianBlur(mask, (blur_size, blur_size), 0)
    return mask


def _compose_adjacent_homographies(pairwise_homographies: Mapping[str, np.ndarray], camera_order: Sequence[str]) -> Dict[str, np.ndarray]:
    reference_camera = str(camera_order[0])
    homographies: Dict[str, np.ndarray] = {reference_camera: np.eye(3, dtype=np.float32)}
    previous_to_reference = np.eye(3, dtype=np.float32)
    for index in range(1, len(camera_order)):
        source_camera = str(camera_order[index])
        target_camera = str(camera_order[index - 1])
        pair_key = f"{source_camera}__to__{target_camera}"
        previous_to_reference = previous_to_reference @ pairwise_homographies[pair_key]
        homographies[source_camera] = previous_to_reference.copy()
    return homographies


def _center_homographies_on_canvas(
    homographies: Mapping[str, np.ndarray],
    image_shapes: Mapping[str, Sequence[int]],
    canvas_width: int,
    canvas_height: int,
) -> Dict[str, np.ndarray]:
    all_corners = []
    for camera_id, matrix in homographies.items():
        height, width = image_shapes[camera_id][:2]
        corners = np.float32([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]]).reshape(-1, 1, 2)
        warped = cv2.perspectiveTransform(corners, matrix).reshape(-1, 2)
        all_corners.append(warped)
    merged = np.vstack(all_corners)
    min_corner = merged.min(axis=0)
    max_corner = merged.max(axis=0)
    current_center = (min_corner + max_corner) / 2.0
    target_center = np.array([canvas_width / 2.0, canvas_height / 2.0], dtype=np.float32)
    dx, dy = target_center - current_center
    translation = np.array([[1.0, 0.0, dx], [0.0, 1.0, dy], [0.0, 0.0, 1.0]], dtype=np.float32)
    return {camera_id: translation @ matrix for camera_id, matrix in homographies.items()}


def build_camera_graph(images, masks=None, config=None, image_shapes=None, debug_dir=None):
    """Build a camera graph from adjacent automatic pairwise homographies."""
    if config is None or image_shapes is None:
        raise ConfigError("Automatic graph construction requires config and image shapes.")
    homography_cfg = config.get("homography", {})
    camera_order = homography_cfg.get("camera_order") or list(images.keys())
    if len(camera_order) < 2:
        raise ConfigError("Automatic homography mode requires at least two ordered cameras.")
    method = str(homography_cfg.get("feature_method", "sift"))
    nfeatures = int(homography_cfg.get("nfeatures", 4000))
    ratio_test = float(homography_cfg.get("ratio_test", 0.75))
    ransac_thresh = float(homography_cfg.get("ransac_reproj_threshold", 4.0))
    min_matches = int(homography_cfg.get("min_matches", 40))
    min_inliers = int(homography_cfg.get("min_inliers", 25))
    use_building_mask = bool(homography_cfg.get("use_building_mask_for_features", True))
    confidence_cfg = homography_cfg.get("confidence", {})
    use_match_support_mask = bool(confidence_cfg.get("use_match_support_mask", True))
    support_dilate = int(confidence_cfg.get("support_mask_dilate", 61))
    support_blur = int(confidence_cfg.get("support_mask_blur", 41))
    min_inlier_ratio = float(confidence_cfg.get("min_inlier_ratio", 0.45))
    warp_model_type = str(homography_cfg.get("warp_model", "global"))

    pairwise_homographies: Dict[str, np.ndarray] = {}
    pair_support_masks: Dict[str, np.ndarray] = {}
    pair_reports: Dict[str, dict[str, float | int | str]] = {}
    if debug_dir is not None:
        ensure_dir(debug_dir)
    for index in range(1, len(camera_order)):
        source_camera = str(camera_order[index])
        target_camera = str(camera_order[index - 1])
        image_source = images[source_camera]
        image_target = images[target_camera]
        source_mask = None
        target_mask = None
        if use_building_mask and masks is not None:
            if masks.get(source_camera) is not None:
                source_mask = cv2.bitwise_not(masks[source_camera])
            if masks.get(target_camera) is not None:
                target_mask = cv2.bitwise_not(masks[target_camera])
        match_result = match_features(
            image_source,
            image_target,
            mask_a=source_mask,
            mask_b=target_mask,
            method=method,
            nfeatures=nfeatures,
            ratio_test=ratio_test,
        )
        matrix, inlier_mask = estimate_pairwise_homography(
            match_result["keypoints_a"],
            match_result["keypoints_b"],
            match_result["matches"],
            ransac_reproj_threshold=ransac_thresh,
            min_matches=min_matches,
            min_inliers=min_inliers,
        )
        inlier_count = int(inlier_mask.sum())
        match_count = int(len(match_result["matches"]))
        inlier_ratio = float(inlier_count / max(match_count, 1))
        if inlier_ratio < min_inlier_ratio:
            raise ConfigError(
                f"Low inlier ratio for {source_camera}->{target_camera}: {inlier_ratio:.3f} < {min_inlier_ratio:.3f}"
            )
        inlier_src_points = np.float32(
            [match_result["keypoints_a"][m.queryIdx].pt for m, keep in zip(match_result["matches"], inlier_mask) if keep]
        )
        support_mask = _make_support_mask(
            image_source.shape[:2],
            inlier_src_points,
            support_dilate,
            support_blur,
        ) if use_match_support_mask else np.full(image_source.shape[:2], 255, dtype=np.uint8)
        warp_model = build_warp_model(
            warp_model_type,
            matrix.astype(np.float32),
            config=homography_cfg,
            source_points=inlier_src_points,
            target_points=np.float32(
                [match_result["keypoints_b"][m.trainIdx].pt for m, keep in zip(match_result["matches"], inlier_mask) if keep]
            ),
        )
        pair_key = f"{source_camera}__to__{target_camera}"
        pairwise_homographies[pair_key] = warp_model.global_homography
        pair_support_masks[pair_key] = support_mask
        pair_reports[pair_key] = {
            "feature_method": str(match_result["method"]),
            "matches": match_count,
            "inliers": inlier_count,
            "inlier_ratio": inlier_ratio,
            "warp_model": warp_model.model_type,
        }
        pair_reports[pair_key]["source_camera"] = source_camera
        pair_reports[pair_key]["target_camera"] = target_camera
        if debug_dir is not None:
            debug_image = _draw_match_visualization(
                image_source,
                image_target,
                match_result["keypoints_a"],
                match_result["keypoints_b"],
                match_result["matches"],
                inlier_mask,
            )
            save_image(debug_dir / f"{pair_key}_matches.jpg", debug_image)
            save_image(debug_dir / f"{pair_key}_support_mask.png", support_mask)

    homographies = _compose_adjacent_homographies(pairwise_homographies, camera_order)
    homographies = _center_homographies_on_canvas(
        homographies,
        image_shapes,
        int(config["canvas"]["width"]),
        int(config["canvas"]["height"]),
    )
    confidence_masks: Dict[str, np.ndarray] = {
        str(camera_order[0]): np.full(image_shapes[str(camera_order[0])][:2], 255, dtype=np.uint8)
    }
    for index in range(1, len(camera_order)):
        source_camera = str(camera_order[index])
        target_camera = str(camera_order[index - 1])
        pair_key = f"{source_camera}__to__{target_camera}"
        confidence_masks[source_camera] = pair_support_masks[pair_key]
    pair_models: Dict[str, dict[str, Any]] = {}
    if warp_model_type == "apap":
        apap_cfg = homography_cfg.get("apap", {})
        for index in range(1, len(camera_order)):
            source_camera = str(camera_order[index])
            target_camera = str(camera_order[index - 1])
            pair_key = f"{source_camera}__to__{target_camera}"
            report = pair_reports[pair_key]
            if report["warp_model"] != "apap":
                continue
            payload_matches = match_features(
                images[source_camera],
                images[target_camera],
                mask_a=cv2.bitwise_not(masks[source_camera]) if use_building_mask and masks and masks.get(source_camera) is not None else None,
                mask_b=cv2.bitwise_not(masks[target_camera]) if use_building_mask and masks and masks.get(target_camera) is not None else None,
                method=method,
                nfeatures=nfeatures,
                ratio_test=ratio_test,
            )
            _, inlier_mask = estimate_pairwise_homography(
                payload_matches["keypoints_a"],
                payload_matches["keypoints_b"],
                payload_matches["matches"],
                ransac_reproj_threshold=ransac_thresh,
                min_matches=min_matches,
                min_inliers=min_inliers,
            )
            src_inliers = np.float32(
                [payload_matches["keypoints_a"][m.queryIdx].pt for m, keep in zip(payload_matches["matches"], inlier_mask) if keep]
            )
            dst_inliers = np.float32(
                [payload_matches["keypoints_b"][m.trainIdx].pt for m, keep in zip(payload_matches["matches"], inlier_mask) if keep]
            )
            apap_model = build_apap_model(
                src_inliers,
                dst_inliers,
                images[target_camera].shape[:2],
                pairwise_homographies[pair_key],
                apap_cfg,
            )
            pair_models[pair_key] = {
                "source_camera": source_camera,
                "target_camera": target_camera,
                "source_points": src_inliers.tolist(),
                "target_points": dst_inliers.tolist(),
                "apap_config": dict(apap_cfg),
                "global_homography": pairwise_homographies[pair_key].tolist(),
            }

    return {
        "homographies": homographies,
        "confidence_masks": confidence_masks,
        "pair_reports": pair_reports,
        "pair_models": pair_models,
    }


def _point_spread_area(points: np.ndarray) -> float:
    if len(points) < 3:
        return 0.0
    hull = cv2.convexHull(points.reshape(-1, 1, 2))
    return float(cv2.contourArea(hull))


def build_reference_homographies(
    control_points_data: Mapping[str, Any],
    config: Mapping[str, Any],
    image_shapes: Mapping[str, Sequence[int]],
    building_masks: Optional[Mapping[str, Optional[np.ndarray]]] = None,
) -> Dict[str, np.ndarray]:
    metadata = control_points_data.get("metadata", {})
    cameras = control_points_data.get("cameras", {})
    reference_camera = metadata.get("reference_camera")
    if not reference_camera:
        raise ConfigError("Reference-image annotation mode requires metadata.reference_camera.")
    if reference_camera not in image_shapes:
        raise ConfigError(f"Reference camera image shape not found: {reference_camera}")

    ref_height, ref_width = image_shapes[reference_camera][:2]
    canvas_cfg = config.get("canvas", {})
    anchor_cfg = canvas_cfg.get("building_anchor", {})
    if anchor_cfg.get("enabled", False):
        anchor_point = np.asarray(anchor_cfg.get("point", [canvas_cfg.get("width", ref_width) / 2, canvas_cfg.get("height", ref_height) / 2]), dtype=np.float32)
        ref_mask = building_masks.get(reference_camera) if building_masks else None
        centroid = _mask_centroid(ref_mask) if ref_mask is not None else None
        if centroid is None:
            centroid = np.array([ref_width / 2.0, ref_height / 2.0], dtype=np.float32)
    else:
        anchor_point = np.array([canvas_cfg.get("width", ref_width) / 2.0, canvas_cfg.get("height", ref_height) / 2.0], dtype=np.float32)
        centroid = np.array([ref_width / 2.0, ref_height / 2.0], dtype=np.float32)

    tx = float(anchor_point[0] - centroid[0])
    ty = float(anchor_point[1] - centroid[1])
    reference_to_canvas = np.array([[1.0, 0.0, tx], [0.0, 1.0, ty], [0.0, 0.0, 1.0]], dtype=np.float32)

    homographies: Dict[str, np.ndarray] = {reference_camera: reference_to_canvas}
    for camera_id, payload in cameras.items():
        if camera_id == reference_camera:
            continue
        src_points = payload.get("src_points", [])
        ref_points = payload.get("ref_points", payload.get("dst_points", []))
        pairwise = compute_homography(src_points, ref_points)
        homographies[camera_id] = reference_to_canvas @ pairwise
    return homographies


def build_adjacent_pair_homographies(
    control_points_data: Mapping[str, Any],
    config: Mapping[str, Any],
    image_shapes: Mapping[str, Sequence[int]],
) -> Dict[str, np.ndarray]:
    metadata = control_points_data.get("metadata", {})
    pairs = control_points_data.get("pairs", {})
    reference_camera = metadata.get("reference_camera")
    camera_order = metadata.get("camera_order")
    if not reference_camera or not camera_order:
        raise ConfigError("Adjacent-pair annotation requires metadata.reference_camera and metadata.camera_order.")
    if reference_camera != camera_order[0]:
        raise ConfigError("Adjacent-pair annotation expects reference_camera to be the first camera in camera_order.")
    if reference_camera not in image_shapes:
        raise ConfigError(f"Reference camera image shape not found: {reference_camera}")

    ref_height, ref_width = image_shapes[reference_camera][:2]
    canvas_cfg = config.get("canvas", {})
    tx = float(canvas_cfg.get("width", ref_width) / 2.0 - ref_width / 2.0)
    ty = float(canvas_cfg.get("height", ref_height) / 2.0 - ref_height / 2.0)
    reference_to_canvas = np.array([[1.0, 0.0, tx], [0.0, 1.0, ty], [0.0, 0.0, 1.0]], dtype=np.float32)

    homographies: Dict[str, np.ndarray] = {reference_camera: reference_to_canvas}
    previous_to_reference = np.eye(3, dtype=np.float32)
    for index in range(1, len(camera_order)):
        source_camera = str(camera_order[index])
        target_camera = str(camera_order[index - 1])
        pair_key = f"{source_camera}__to__{target_camera}"
        payload = pairs.get(pair_key)
        if not payload:
            raise ConfigError(f"Missing adjacent pair annotation: {pair_key}")
        pairwise = compute_homography(payload.get("src_points", []), payload.get("dst_points", []))
        previous_to_reference = previous_to_reference @ pairwise
        homographies[source_camera] = reference_to_canvas @ previous_to_reference
    return homographies


def save_homographies(path: Path, homographies: Mapping[str, np.ndarray]) -> None:
    payload = {camera_id: matrix.tolist() for camera_id, matrix in homographies.items()}
    save_yaml(path, payload)


def load_homographies(path: Path) -> Dict[str, np.ndarray]:
    data = load_yaml(path)
    if not isinstance(data, dict) or not data:
        raise ConfigError(f"Homography file is empty: {path}")
    result: Dict[str, np.ndarray] = {}
    for camera_id, matrix in data.items():
        arr = np.asarray(matrix, dtype=np.float32)
        if arr.shape != (3, 3):
            raise ConfigError(f"Invalid homography shape for {camera_id}: {arr.shape}")
        result[camera_id] = arr
    return result


def ensure_homographies(
    control_points_path: Path,
    homography_path: Path,
    compute_if_missing: bool,
    config: Optional[Mapping[str, Any]] = None,
    image_shapes: Optional[Mapping[str, Sequence[int]]] = None,
    building_masks: Optional[Mapping[str, Optional[np.ndarray]]] = None,
) -> Dict[str, np.ndarray]:
    if config is not None and config.get("homography", {}).get("mode", "manual") == "auto":
        raise ConfigError("Automatic homography mode should be computed from images, not control-point files.")
    if homography_path.exists():
        if not control_points_path.exists():
            return load_homographies(homography_path)
        if homography_path.stat().st_mtime >= control_points_path.stat().st_mtime:
            return load_homographies(homography_path)
    if not compute_if_missing:
        raise ConfigError(f"Homography file not found: {homography_path}")
    control_points = load_yaml(control_points_path)
    metadata = control_points.get("metadata", {}) if isinstance(control_points, dict) else {}
    if metadata.get("annotation_mode") == "reference_image":
        if config is None or image_shapes is None:
            raise ConfigError("Reference-image homography computation requires config and image shapes.")
        homographies = build_reference_homographies(control_points, config, image_shapes, building_masks)
    elif metadata.get("annotation_mode") == "adjacent_pair":
        if config is None or image_shapes is None:
            raise ConfigError("Adjacent-pair homography computation requires config and image shapes.")
        homographies = build_adjacent_pair_homographies(control_points, config, image_shapes)
    else:
        flat_control_points = load_control_points(control_points_path)
        homographies = compute_homographies_from_points(flat_control_points)
    save_homographies(homography_path, homographies)
    return homographies


def _mask_centroid(mask: Optional[np.ndarray]) -> Optional[np.ndarray]:
    if mask is None:
        return None
    moments = cv2.moments((mask > 0).astype(np.uint8))
    if moments["m00"] == 0:
        return None
    return np.array([moments["m10"] / moments["m00"], moments["m01"] / moments["m00"]], dtype=np.float32)


def anchor_homography_to_building_center(
    homography: np.ndarray,
    building_mask: Optional[np.ndarray],
    anchor_point: Sequence[float],
    canvas_size: Sequence[int],
) -> np.ndarray:
    if building_mask is None:
        return homography
    width, height = int(canvas_size[0]), int(canvas_size[1])
    warped_mask = cv2.warpPerspective(building_mask, homography, (width, height))
    centroid = _mask_centroid(warped_mask)
    if centroid is None:
        return homography
    target = np.asarray(anchor_point, dtype=np.float32)
    dx, dy = float(target[0] - centroid[0]), float(target[1] - centroid[1])
    translation = np.array([[1.0, 0.0, dx], [0.0, 1.0, dy], [0.0, 0.0, 1.0]], dtype=np.float32)
    return translation @ homography


def optimize_global_layout():
    """V2 placeholder for global layout optimization."""
    raise NotImplementedError("Global layout optimization is reserved for V2.")
