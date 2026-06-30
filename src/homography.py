from __future__ import annotations

import math
from collections import deque
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

import cv2
import numpy as np

from apap import build_apap_model, build_warp_model
from utils import ConfigError, ensure_dir, load_yaml, save_image, save_yaml, warning

HOMOGRAPHY_METADATA_KEY = "__meta__"


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
    return matrix.astype(np.float32)


def compute_homographies_from_points(control_points: Mapping[str, Any]) -> Dict[str, np.ndarray]:
    homographies: Dict[str, np.ndarray] = {}
    for camera_id, points in control_points.items():
        src_points = points.get("src_points", [])
        dst_points = points.get("dst_points", [])
        homographies[camera_id] = compute_homography(src_points, dst_points)
    return homographies


def _create_classical_feature_detector(method: str, nfeatures: int):
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
    detector, _, resolved_method = _create_classical_feature_detector(method, nfeatures)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    keypoints, descriptors = detector.detectAndCompute(gray, mask)
    return keypoints or [], descriptors, resolved_method


def _to_gray_tensor(image: np.ndarray):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    return gray.astype(np.float32) / 255.0


def _try_match_superpoint_lightglue(image_a, image_b, mask_a=None, mask_b=None, max_keypoints=4096):
    try:
        import torch
        from lightglue import LightGlue, SuperPoint
        from lightglue.utils import rbd
    except Exception as exc:
        raise ConfigError(f"SuperPoint+LightGlue dependencies unavailable: {exc}") from exc

    device = "cuda" if torch.cuda.is_available() else "cpu"
    extractor = SuperPoint(max_num_keypoints=max_keypoints).eval().to(device)
    matcher = LightGlue(features="superpoint").eval().to(device)

    tensor_a = torch.from_numpy(_to_gray_tensor(image_a))[None, None].to(device)
    tensor_b = torch.from_numpy(_to_gray_tensor(image_b))[None, None].to(device)
    feats_a = extractor.extract(tensor_a)
    feats_b = extractor.extract(tensor_b)

    if mask_a is not None:
        keep = mask_a[np.round(feats_a["keypoints"][0, :, 1].cpu().numpy()).astype(int), np.round(feats_a["keypoints"][0, :, 0].cpu().numpy()).astype(int)] > 0
        keep_tensor = torch.from_numpy(keep).to(device=device, dtype=torch.bool)
        feats_a = {key: value[:, keep_tensor] if value.ndim >= 3 else value for key, value in feats_a.items()}
    if mask_b is not None:
        keep = mask_b[np.round(feats_b["keypoints"][0, :, 1].cpu().numpy()).astype(int), np.round(feats_b["keypoints"][0, :, 0].cpu().numpy()).astype(int)] > 0
        keep_tensor = torch.from_numpy(keep).to(device=device, dtype=torch.bool)
        feats_b = {key: value[:, keep_tensor] if value.ndim >= 3 else value for key, value in feats_b.items()}

    matches01 = matcher({"image0": feats_a, "image1": feats_b})
    feats_a, feats_b, matches01 = [rbd(item) for item in (feats_a, feats_b, matches01)]
    matches = matches01["matches"].detach().cpu().numpy()
    points_a = feats_a["keypoints"].detach().cpu().numpy()
    points_b = feats_b["keypoints"].detach().cpu().numpy()

    keypoints_a = [cv2.KeyPoint(float(x), float(y), 1.0) for x, y in points_a]
    keypoints_b = [cv2.KeyPoint(float(x), float(y), 1.0) for x, y in points_b]
    cv_matches = [
        cv2.DMatch(_queryIdx=int(index_a), _trainIdx=int(index_b), _distance=0.0)
        for index_a, index_b in matches
    ]
    return {
        "method": "superpoint_lightglue",
        "keypoints_a": keypoints_a,
        "keypoints_b": keypoints_b,
        "matches": cv_matches,
    }


def _try_match_loftr(image_a, image_b, mask_a=None, mask_b=None):
    try:
        import torch
        import kornia as K
        import kornia.feature as KF
    except Exception as exc:
        raise ConfigError(f"LoFTR dependencies unavailable: {exc}") from exc

    device = "cuda" if torch.cuda.is_available() else "cpu"
    matcher = KF.LoFTR(pretrained="outdoor").eval().to(device)
    tensor_a = torch.from_numpy(_to_gray_tensor(image_a))[None, None].to(device)
    tensor_b = torch.from_numpy(_to_gray_tensor(image_b))[None, None].to(device)
    with torch.inference_mode():
        result = matcher({"image0": tensor_a, "image1": tensor_b})
    points_a = result["keypoints0"].detach().cpu().numpy()
    points_b = result["keypoints1"].detach().cpu().numpy()

    if mask_a is not None:
        keep_a = mask_a[np.clip(np.round(points_a[:, 1]).astype(int), 0, mask_a.shape[0] - 1), np.clip(np.round(points_a[:, 0]).astype(int), 0, mask_a.shape[1] - 1)] > 0
    else:
        keep_a = np.ones(len(points_a), dtype=bool)
    if mask_b is not None:
        keep_b = mask_b[np.clip(np.round(points_b[:, 1]).astype(int), 0, mask_b.shape[0] - 1), np.clip(np.round(points_b[:, 0]).astype(int), 0, mask_b.shape[1] - 1)] > 0
    else:
        keep_b = np.ones(len(points_b), dtype=bool)
    keep = keep_a & keep_b
    points_a = points_a[keep]
    points_b = points_b[keep]

    keypoints_a = [cv2.KeyPoint(float(x), float(y), 1.0) for x, y in points_a]
    keypoints_b = [cv2.KeyPoint(float(x), float(y), 1.0) for x, y in points_b]
    cv_matches = [
        cv2.DMatch(_queryIdx=index, _trainIdx=index, _distance=0.0)
        for index in range(len(points_a))
    ]
    return {
        "method": "loftr",
        "keypoints_a": keypoints_a,
        "keypoints_b": keypoints_b,
        "matches": cv_matches,
    }


def match_features(image_a, image_b, mask_a=None, mask_b=None, method="sift", nfeatures=4000, ratio_test=0.75):
    normalized = str(method).lower()
    if normalized == "superpoint_lightglue":
        try:
            return _try_match_superpoint_lightglue(image_a, image_b, mask_a, mask_b, max_keypoints=nfeatures)
        except ConfigError as exc:
            warning(f"{exc}; falling back to SIFT.")
            normalized = "sift"
    if normalized == "loftr":
        try:
            return _try_match_loftr(image_a, image_b, mask_a, mask_b)
        except ConfigError as exc:
            warning(f"{exc}; falling back to SIFT.")
            normalized = "sift"

    keypoints_a, descriptors_a, resolved_method = detect_features(image_a, mask_a, normalized, nfeatures)
    keypoints_b, descriptors_b, _ = detect_features(image_b, mask_b, resolved_method, nfeatures)
    if descriptors_a is None or descriptors_b is None:
        raise ConfigError("Failed to compute descriptors for automatic matching.")
    _, norm_type, _ = _create_classical_feature_detector(resolved_method, nfeatures)
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
    return matrix.astype(np.float32), inlier_mask.ravel().astype(bool)


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


def _point_spread_area(points: np.ndarray) -> float:
    if len(points) < 3:
        return 0.0
    hull = cv2.convexHull(points.reshape(-1, 1, 2))
    return float(cv2.contourArea(hull))


def _compute_pair_score(match_count: int, inlier_count: int, inlier_ratio: float, support_mask: np.ndarray) -> float:
    support_area = float(np.count_nonzero(support_mask)) / float(max(support_mask.size, 1))
    return math.log1p(match_count) * max(inlier_ratio, 1e-6) * (0.25 + support_area) * math.log1p(inlier_count)


def _resolve_camera_pairs(camera_ids: Sequence[str], graph_mode: str) -> list[tuple[str, str]]:
    if graph_mode == "adjacent_chain":
        return [(str(camera_ids[index]), str(camera_ids[index - 1])) for index in range(1, len(camera_ids))]
    pairs: list[tuple[str, str]] = []
    for src_index in range(len(camera_ids)):
        for dst_index in range(src_index):
            pairs.append((str(camera_ids[src_index]), str(camera_ids[dst_index])))
    return pairs


def _select_reference_camera(camera_ids: Sequence[str], pair_reports: Mapping[str, Mapping[str, Any]], graph_mode: str) -> str:
    if graph_mode == "adjacent_chain":
        return str(camera_ids[0])
    scores = {str(camera_id): 0.0 for camera_id in camera_ids}
    for report in pair_reports.values():
        source_camera = str(report["source_camera"])
        target_camera = str(report["target_camera"])
        score = float(report.get("score", 0.0))
        scores[source_camera] += score
        scores[target_camera] += score
    return max(scores.items(), key=lambda item: item[1])[0]


def _build_maximum_spanning_tree(camera_ids: Sequence[str], pair_reports: Mapping[str, Mapping[str, Any]], graph_mode: str) -> list[dict[str, Any]]:
    if graph_mode == "adjacent_chain":
        edges = []
        for report in pair_reports.values():
            edges.append(
                {
                    "a": str(report["source_camera"]),
                    "b": str(report["target_camera"]),
                    "score": float(report["score"]),
                }
            )
        return edges

    parent = {str(camera_id): str(camera_id) for camera_id in camera_ids}

    def find(node: str) -> str:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    def union(a: str, b: str) -> bool:
        root_a = find(a)
        root_b = find(b)
        if root_a == root_b:
            return False
        parent[root_b] = root_a
        return True

    candidates = sorted(
        (
            {
                "a": str(report["source_camera"]),
                "b": str(report["target_camera"]),
                "score": float(report["score"]),
            }
            for report in pair_reports.values()
        ),
        key=lambda item: item["score"],
        reverse=True,
    )
    selected: list[dict[str, Any]] = []
    for edge in candidates:
        if union(edge["a"], edge["b"]):
            selected.append(edge)
        if len(selected) == len(camera_ids) - 1:
            break
    if len(selected) != len(camera_ids) - 1:
        raise ConfigError("Camera overlap graph is disconnected; cannot build a global layout.")
    return selected


def _compose_global_homographies(
    camera_ids: Sequence[str],
    reference_camera: str,
    tree_edges: Sequence[Mapping[str, Any]],
    directed_homographies: Mapping[tuple[str, str], np.ndarray],
) -> tuple[Dict[str, np.ndarray], Dict[str, str]]:
    adjacency: Dict[str, list[str]] = {str(camera_id): [] for camera_id in camera_ids}
    for edge in tree_edges:
        a = str(edge["a"])
        b = str(edge["b"])
        adjacency[a].append(b)
        adjacency[b].append(a)

    homographies: Dict[str, np.ndarray] = {reference_camera: np.eye(3, dtype=np.float32)}
    parent_map: Dict[str, str] = {}
    queue = deque([reference_camera])
    while queue:
        current = queue.popleft()
        for neighbor in adjacency[current]:
            if neighbor in homographies:
                continue
            if (neighbor, current) in directed_homographies:
                edge_h = directed_homographies[(neighbor, current)]
            elif (current, neighbor) in directed_homographies:
                edge_h = np.linalg.inv(directed_homographies[(current, neighbor)]).astype(np.float32)
            else:
                raise ConfigError(f"Missing homography edge between {neighbor} and {current}.")
            homographies[neighbor] = homographies[current] @ edge_h
            parent_map[neighbor] = current
            queue.append(neighbor)

    if len(homographies) != len(camera_ids):
        raise ConfigError("Failed to compose homographies for all cameras.")
    return homographies, parent_map


def _fit_canvas_to_homographies(
    homographies: Mapping[str, np.ndarray],
    image_shapes: Mapping[str, Sequence[int]],
    canvas_cfg: Mapping[str, Any],
) -> tuple[Dict[str, np.ndarray], dict[str, Any]]:
    mode = str(canvas_cfg.get("mode", "fixed")).lower()
    if mode not in {"fixed", "auto_fit"}:
        raise ConfigError(f"Unsupported canvas.mode: {mode}")

    all_corners = []
    for camera_id, matrix in homographies.items():
        height, width = image_shapes[camera_id][:2]
        corners = np.float32([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]]).reshape(-1, 1, 2)
        warped = cv2.perspectiveTransform(corners, matrix).reshape(-1, 2)
        all_corners.append(warped)
    merged = np.vstack(all_corners)
    min_corner = merged.min(axis=0)
    max_corner = merged.max(axis=0)

    if mode == "fixed":
        canvas_width = int(canvas_cfg["width"])
        canvas_height = int(canvas_cfg["height"])
        current_center = (min_corner + max_corner) / 2.0
        target_center = np.array([canvas_width / 2.0, canvas_height / 2.0], dtype=np.float32)
        dx, dy = target_center - current_center
    else:
        margin = int(canvas_cfg.get("margin_px", 64))
        canvas_width = int(math.ceil(float(max_corner[0] - min_corner[0]) + margin * 2))
        canvas_height = int(math.ceil(float(max_corner[1] - min_corner[1]) + margin * 2))
        dx = float(margin - min_corner[0])
        dy = float(margin - min_corner[1])

    translation = np.array([[1.0, 0.0, dx], [0.0, 1.0, dy], [0.0, 0.0, 1.0]], dtype=np.float32)
    adjusted = {camera_id: translation @ matrix for camera_id, matrix in homographies.items()}
    metadata = {
        "canvas_mode": mode,
        "canvas_width": int(canvas_width),
        "canvas_height": int(canvas_height),
        "translation": translation.tolist(),
        "bounds_before_translation": {
            "min": [float(min_corner[0]), float(min_corner[1])],
            "max": [float(max_corner[0]), float(max_corner[1])],
        },
    }
    return adjusted, metadata


def build_camera_graph(images, masks=None, config=None, image_shapes=None, debug_dir=None):
    if config is None or image_shapes is None:
        raise ConfigError("Automatic graph construction requires config and image shapes.")
    homography_cfg = config.get("homography", {})
    camera_order = [str(camera_id) for camera_id in (homography_cfg.get("camera_order") or list(images.keys()))]
    if len(camera_order) < 2:
        raise ConfigError("Automatic homography mode requires at least two ordered cameras.")

    method = str(homography_cfg.get("feature_method", "sift"))
    graph_mode = str(homography_cfg.get("graph_mode", "adjacent_chain")).lower()
    layout_optimization = str(homography_cfg.get("layout_optimization", "global_homography_graph")).lower()
    nfeatures = int(homography_cfg.get("nfeatures", 4000))
    ratio_test = float(homography_cfg.get("ratio_test", 0.75))
    ransac_thresh = float(homography_cfg.get("ransac_reproj_threshold", 4.0))
    min_matches = int(homography_cfg.get("min_matches", 40))
    min_inliers = int(homography_cfg.get("min_inliers", 25))
    use_building_mask = bool(homography_cfg.get("use_building_mask_for_features", True))
    use_ground_priority = bool(config.get("masks", {}).get("ground_priority", False))
    confidence_cfg = homography_cfg.get("confidence", {})
    use_match_support_mask = bool(confidence_cfg.get("use_match_support_mask", True))
    support_dilate = int(confidence_cfg.get("support_mask_dilate", 61))
    support_blur = int(confidence_cfg.get("support_mask_blur", 41))
    min_inlier_ratio = float(confidence_cfg.get("min_inlier_ratio", 0.45))
    warp_model_type = str(homography_cfg.get("warp_model", "global"))

    if graph_mode not in {"adjacent_chain", "overlap_graph"}:
        raise ConfigError(f"Unsupported homography.graph_mode: {graph_mode}")
    if layout_optimization not in {"none", "global_homography_graph"}:
        raise ConfigError(f"Unsupported homography.layout_optimization: {layout_optimization}")

    pair_reports: Dict[str, dict[str, float | int | str]] = {}
    pair_support_masks: Dict[str, np.ndarray] = {}
    directed_homographies: Dict[tuple[str, str], np.ndarray] = {}
    pair_models: Dict[str, dict[str, Any]] = {}
    if debug_dir is not None:
        ensure_dir(debug_dir)

    for source_camera, target_camera in _resolve_camera_pairs(camera_order, graph_mode):
        image_source = images[source_camera]
        image_target = images[target_camera]
        source_mask = None
        target_mask = None
        if use_building_mask and masks is not None:
            if masks.get(source_camera) is not None:
                source_mask = cv2.bitwise_not(masks[source_camera])
            if masks.get(target_camera) is not None:
                target_mask = cv2.bitwise_not(masks[target_camera])
        if use_ground_priority and source_mask is not None and target_mask is not None:
            source_mask = cv2.erode(source_mask, np.ones((5, 5), dtype=np.uint8))
            target_mask = cv2.erode(target_mask, np.ones((5, 5), dtype=np.uint8))

        try:
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
        except ConfigError as exc:
            if graph_mode == "adjacent_chain":
                raise
            warning(f"Skipping weak overlap edge {source_camera}->{target_camera}: {exc}")
            continue

        inlier_src_points = np.float32(
            [match_result["keypoints_a"][m.queryIdx].pt for m, keep in zip(match_result["matches"], inlier_mask) if keep]
        )
        inlier_dst_points = np.float32(
            [match_result["keypoints_b"][m.trainIdx].pt for m, keep in zip(match_result["matches"], inlier_mask) if keep]
        )
        support_mask = (
            _make_support_mask(image_source.shape[:2], inlier_src_points, support_dilate, support_blur)
            if use_match_support_mask
            else np.full(image_source.shape[:2], 255, dtype=np.uint8)
        )
        warp_model = build_warp_model(
            warp_model_type,
            matrix.astype(np.float32),
            config=homography_cfg,
            source_points=inlier_src_points,
            target_points=inlier_dst_points,
        )
        pair_key = f"{source_camera}__to__{target_camera}"
        score = _compute_pair_score(match_count, inlier_count, inlier_ratio, support_mask)
        directed_homographies[(source_camera, target_camera)] = warp_model.global_homography
        pair_support_masks[pair_key] = support_mask
        pair_reports[pair_key] = {
            "feature_method": str(match_result["method"]),
            "matches": match_count,
            "inliers": inlier_count,
            "inlier_ratio": inlier_ratio,
            "warp_model": warp_model.model_type,
            "source_camera": source_camera,
            "target_camera": target_camera,
            "score": float(score),
            "support_area_ratio": float(np.count_nonzero(support_mask) / max(support_mask.size, 1)),
        }

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

        if warp_model.model_type == "apap":
            pair_models[pair_key] = {
                "source_camera": source_camera,
                "target_camera": target_camera,
                "source_points": inlier_src_points.tolist(),
                "target_points": inlier_dst_points.tolist(),
                "apap_config": dict(homography_cfg.get("apap", {})),
                "global_homography": warp_model.global_homography.tolist(),
            }

    reference_camera = _select_reference_camera(camera_order, pair_reports, graph_mode)
    tree_edges = _build_maximum_spanning_tree(camera_order, pair_reports, graph_mode)
    homographies, parent_map = _compose_global_homographies(camera_order, reference_camera, tree_edges, directed_homographies)
    if layout_optimization == "global_homography_graph":
        homographies, canvas_metadata = _fit_canvas_to_homographies(homographies, image_shapes, config["canvas"])
    else:
        homographies, canvas_metadata = _fit_canvas_to_homographies(homographies, image_shapes, {**config["canvas"], "mode": "fixed"})

    confidence_masks: Dict[str, np.ndarray] = {}
    for camera_id in camera_order:
        parent = parent_map.get(camera_id)
        if parent is None:
            confidence_masks[camera_id] = np.full(image_shapes[camera_id][:2], 255, dtype=np.uint8)
            continue
        pair_key = f"{camera_id}__to__{parent}"
        inverse_pair_key = f"{parent}__to__{camera_id}"
        if pair_key in pair_support_masks:
            confidence_masks[camera_id] = pair_support_masks[pair_key]
        elif inverse_pair_key in pair_support_masks:
            confidence_masks[camera_id] = np.full(image_shapes[camera_id][:2], 255, dtype=np.uint8)
        else:
            confidence_masks[camera_id] = np.full(image_shapes[camera_id][:2], 255, dtype=np.uint8)

    selected_pair_models: Dict[str, dict[str, Any]] = {}
    for child, parent in parent_map.items():
        pair_key = f"{child}__to__{parent}"
        if pair_key in pair_models:
            selected_pair_models[pair_key] = pair_models[pair_key]

    return {
        "homographies": homographies,
        "confidence_masks": confidence_masks,
        "pair_reports": pair_reports,
        "pair_models": selected_pair_models,
        "reference_camera": reference_camera,
        "parent_map": parent_map,
        "canvas_metadata": canvas_metadata,
        "layout_graph": {
            "mode": graph_mode,
            "optimization": layout_optimization,
            "tree_edges": list(tree_edges),
        },
    }


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


def save_homographies(path: Path, homographies: Mapping[str, np.ndarray], metadata: Optional[Mapping[str, Any]] = None) -> None:
    payload: Dict[str, Any] = {camera_id: matrix.tolist() for camera_id, matrix in homographies.items()}
    if metadata:
        payload[HOMOGRAPHY_METADATA_KEY] = dict(metadata)
    save_yaml(path, payload)


def load_homography_bundle(path: Path) -> tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    data = load_yaml(path)
    if not isinstance(data, dict) or not data:
        raise ConfigError(f"Homography file is empty: {path}")
    metadata = data.get(HOMOGRAPHY_METADATA_KEY, {})
    result: Dict[str, np.ndarray] = {}
    for camera_id, matrix in data.items():
        if camera_id == HOMOGRAPHY_METADATA_KEY:
            continue
        arr = np.asarray(matrix, dtype=np.float32)
        if arr.shape != (3, 3):
            raise ConfigError(f"Invalid homography shape for {camera_id}: {arr.shape}")
        result[camera_id] = arr
    return result, metadata if isinstance(metadata, dict) else {}


def load_homographies(path: Path) -> Dict[str, np.ndarray]:
    homographies, _ = load_homography_bundle(path)
    return homographies


def ensure_homography_bundle(
    control_points_path: Path,
    homography_path: Path,
    compute_if_missing: bool,
    config: Optional[Mapping[str, Any]] = None,
    image_shapes: Optional[Mapping[str, Sequence[int]]] = None,
    building_masks: Optional[Mapping[str, Optional[np.ndarray]]] = None,
) -> tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    if config is not None and config.get("homography", {}).get("mode", "auto") == "auto":
        raise ConfigError("Automatic homography mode should be computed from images, not control-point files.")
    if homography_path.exists():
        if not control_points_path.exists():
            return load_homography_bundle(homography_path)
        if homography_path.stat().st_mtime >= control_points_path.stat().st_mtime:
            return load_homography_bundle(homography_path)
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
    homographies, bundle_metadata = _fit_canvas_to_homographies(homographies, image_shapes or {}, config.get("canvas", {}) if config else {"mode": "fixed"})
    save_homographies(homography_path, homographies, bundle_metadata)
    return homographies, bundle_metadata


def ensure_homographies(
    control_points_path: Path,
    homography_path: Path,
    compute_if_missing: bool,
    config: Optional[Mapping[str, Any]] = None,
    image_shapes: Optional[Mapping[str, Sequence[int]]] = None,
    building_masks: Optional[Mapping[str, Optional[np.ndarray]]] = None,
) -> Dict[str, np.ndarray]:
    homographies, _ = ensure_homography_bundle(
        control_points_path,
        homography_path,
        compute_if_missing,
        config=config,
        image_shapes=image_shapes,
        building_masks=building_masks,
    )
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
    raise NotImplementedError("Global layout optimization is reserved for a later iteration.")
