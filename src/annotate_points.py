from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from utils import (
    compute_display_scale,
    get_input_paths,
    info,
    load_config,
    parse_args,
    save_yaml,
    load_image,
)


SOURCE_WINDOW = "Source Image"
REFERENCE_WINDOW = "Reference Image"


@dataclass
class CameraPoints:
    src_points: List[List[int]] = field(default_factory=list)
    dst_points: List[List[int]] = field(default_factory=list)


@dataclass
class PairTask:
    source_camera: str
    target_camera: str

    @property
    def key(self) -> str:
        return f"{self.source_camera}__to__{self.target_camera}"


class PointAnnotationSession:
    def __init__(self, config: dict, root: Path) -> None:
        self.config = config
        self.root = root
        self.camera_ids = list(get_input_paths(config, root, mode="image").keys())
        self.image_paths = get_input_paths(config, root, mode="image")
        annotation_cfg = config.get("annotation", {})
        self.annotation_mode = str(annotation_cfg.get("mode", "adjacent_pair"))
        self.reference_camera = str(
            annotation_cfg.get("reference_camera", config.get("blending", {}).get("building_reference_camera", self.camera_ids[0]))
        )
        if self.reference_camera not in self.camera_ids:
            raise ValueError(f"Reference camera not found in image paths: {self.reference_camera}")
        self.tasks = self._build_tasks(annotation_cfg)
        self.points: Dict[str, CameraPoints] = {task.key: CameraPoints() for task in self.tasks}
        self.index = 0
        self.pending_src_point: Optional[Tuple[int, int]] = None
        self.source_scale = 1.0
        self.reference_scale = 1.0
        self.min_points = int(annotation_cfg.get("min_points_per_camera", 4))
        self.max_points = int(annotation_cfg.get("max_points_per_camera", 20))
        self.dragging_src_index: Optional[int] = None
        self.dragging_ref_index: Optional[int] = None
        self.pending_exit_confirm = False

    @property
    def task(self) -> PairTask:
        return self.tasks[self.index]

    @property
    def current_state(self) -> CameraPoints:
        return self.points[self.task.key]

    @property
    def camera_id(self) -> str:
        return self.task.source_camera

    @property
    def target_camera_id(self) -> str:
        return self.task.target_camera

    def _build_tasks(self, annotation_cfg: dict) -> List[PairTask]:
        mode = self.annotation_mode
        if mode == "adjacent_pair":
            order = annotation_cfg.get("camera_order", self.camera_ids)
            if not isinstance(order, list) or len(order) < 2:
                raise ValueError("annotation.camera_order must contain at least two cameras.")
            tasks = []
            for index in range(1, len(order)):
                source_camera = str(order[index])
                target_camera = str(order[index - 1])
                if source_camera not in self.image_paths or target_camera not in self.image_paths:
                    raise ValueError(f"Camera in annotation.camera_order not found in input image_paths: {source_camera}, {target_camera}")
                tasks.append(PairTask(source_camera=source_camera, target_camera=target_camera))
            return tasks
        if mode == "reference_image":
            tasks = []
            for camera_id in self.camera_ids:
                if camera_id == self.reference_camera:
                    continue
                tasks.append(PairTask(source_camera=camera_id, target_camera=self.reference_camera))
            return tasks
        raise ValueError(f"Unsupported annotation mode: {mode}")

    def current_image(self) -> np.ndarray:
        return load_image(self.image_paths[self.camera_id])

    def reference_image(self) -> np.ndarray:
        return load_image(self.image_paths[self.target_camera_id])

    def _status_lines(self) -> List[str]:
        count = len(self.current_state.src_points)
        next_index = count + 1 if self.pending_src_point is None else count + 1
        lines = [
            f"Pair: {self.camera_id} -> {self.target_camera_id} ({self.index + 1}/{len(self.tasks)})",
            f"Pairs: {count}/{self.min_points}+ recommended, max {self.max_points}",
            f"Next pair #: {next_index}",
            "Source click -> Reference click to add a pair",
            "Drag existing points in either window to refine correspondence",
            "N/Enter: finish current camera and go next | S: save progress",
            "U: undo last pair | C: clear current | Q/Esc: exit with save confirm",
        ]
        if self.pending_src_point is not None:
            lines.append(f"Pending source point: {self.pending_src_point}")
        if self.pending_exit_confirm:
            lines.append("Exit? Press Y to save and exit, N to exit without saving, other keys to cancel.")
        return lines

    def _draw_status(self, image: np.ndarray) -> np.ndarray:
        output = image.copy()
        y = 30
        for line in self._status_lines():
            cv2.putText(
                output,
                line,
                (20, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 255),
                2,
            )
            y += 28
        return output

    def draw_source(self) -> np.ndarray:
        image = self.current_image().copy()
        state = self.current_state
        for idx, point in enumerate(state.src_points):
            cv2.circle(image, tuple(point), 6, (0, 255, 0), -1)
            cv2.putText(image, str(idx + 1), tuple(point), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        if self.pending_src_point is not None:
            cv2.circle(image, self.pending_src_point, 6, (0, 255, 255), -1)
        return self._draw_status(image)

    def draw_reference(self) -> np.ndarray:
        canvas = self.reference_image().copy()
        state = self.current_state
        for idx, point in enumerate(state.dst_points):
            cv2.circle(canvas, tuple(point), 6, (255, 255, 0), -1)
            cv2.putText(canvas, str(idx + 1), tuple(point), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        return self._draw_status(canvas)

    def _find_nearby_index(self, points: List[List[int]], x: int, y: int, scale: float, radius: int = 18) -> Optional[int]:
        for idx, point in enumerate(points):
            px = point[0] * scale
            py = point[1] * scale
            if (px - x) ** 2 + (py - y) ** 2 <= radius ** 2:
                return idx
        return None

    def _find_nearby_src_index(self, x: int, y: int, radius: int = 18) -> Optional[int]:
        state = self.current_state
        return self._find_nearby_index(state.src_points, x, y, self.source_scale, radius)

    def _find_nearby_ref_index(self, x: int, y: int, radius: int = 18) -> Optional[int]:
        state = self.current_state
        return self._find_nearby_index(state.dst_points, x, y, self.reference_scale, radius)

    def on_source_click(self, event, x, y, flags, param) -> None:
        _ = flags, param
        state = self.current_state
        mapped_point = (
            int(round(x / self.source_scale)),
            int(round(y / self.source_scale)),
        )
        if event == cv2.EVENT_LBUTTONDOWN:
            nearby_index = self._find_nearby_src_index(x, y)
            if nearby_index is not None:
                self.dragging_src_index = nearby_index
                return
            self.pending_src_point = (
                int(mapped_point[0]),
                int(mapped_point[1]),
            )
        elif event == cv2.EVENT_MOUSEMOVE and self.dragging_src_index is not None:
            state.src_points[self.dragging_src_index] = [int(mapped_point[0]), int(mapped_point[1])]
        elif event == cv2.EVENT_LBUTTONUP:
            self.dragging_src_index = None

    def on_reference_click(self, event, x, y, flags, param) -> None:
        _ = flags, param
        state = self.current_state
        mapped_point = [
            int(round(x / self.reference_scale)),
            int(round(y / self.reference_scale)),
        ]
        if event == cv2.EVENT_LBUTTONDOWN:
            nearby_index = self._find_nearby_ref_index(x, y)
            if nearby_index is not None:
                self.dragging_ref_index = nearby_index
                return
            if self.pending_src_point is not None:
                if len(state.src_points) >= self.max_points:
                    info(f"{self.camera_id} has reached max point pairs: {self.max_points}")
                    self.pending_src_point = None
                    return
                state.src_points.append([int(self.pending_src_point[0]), int(self.pending_src_point[1])])
                state.dst_points.append(mapped_point)
                self.pending_src_point = None
        elif event == cv2.EVENT_MOUSEMOVE and self.dragging_ref_index is not None:
            state.dst_points[self.dragging_ref_index] = mapped_point
        elif event == cv2.EVENT_LBUTTONUP:
            self.dragging_ref_index = None

    def undo(self) -> None:
        state = self.current_state
        if state.src_points:
            state.src_points.pop()
            state.dst_points.pop()
        else:
            self.pending_src_point = None

    def clear(self) -> None:
        self.points[self.camera_id] = CameraPoints()
        self.pending_src_point = None

    def can_finish_current_camera(self) -> bool:
        return len(self.current_state.src_points) >= self.min_points

    def next_camera(self, force: bool = False) -> None:
        if not force and not self.can_finish_current_camera():
            info(f"{self.camera_id} requires at least {self.min_points} point pairs before finishing.")
            return
        self.pending_src_point = None
        next_index = self.index + 1
        if next_index < len(self.tasks):
            self.index = next_index
            return
        info("Already at last pair. Saving progress.")
        self.save()

    def save(self) -> None:
        output = {
            "metadata": {
                "annotation_mode": self.annotation_mode,
                "reference_camera": self.reference_camera,
                "camera_order": [task.target_camera for task in self.tasks[:1]] + [task.source_camera for task in self.tasks],
            },
            "pairs": {},
        }
        for task in self.tasks:
            state = self.points[task.key]
            output["pairs"][task.key] = {
                "source_camera": task.source_camera,
                "target_camera": task.target_camera,
                "src_points": state.src_points,
                "dst_points": state.dst_points,
            }
        output_path = self.root / self.config["homography"]["control_points_file"]
        save_yaml(output_path, output)
        info(f"Saved control points to {output_path}")

    def save_and_finish_current(self) -> None:
        if not self.can_finish_current_camera():
            info(f"{self.camera_id} requires at least {self.min_points} point pairs before finishing.")
            return
        self.save()
        self.next_camera(force=True)

    def handle_exit_confirmation(self, key: int) -> bool:
        if not self.pending_exit_confirm:
            return False
        if key in (ord("y"), ord("Y")):
            self.save()
            return True
        if key in (ord("n"), ord("N")):
            return True
        self.pending_exit_confirm = False
        return False

    def run(self) -> None:
        cv2.namedWindow(SOURCE_WINDOW, cv2.WINDOW_NORMAL)
        cv2.namedWindow(REFERENCE_WINDOW, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(SOURCE_WINDOW, self.on_source_click)
        cv2.setMouseCallback(REFERENCE_WINDOW, self.on_reference_click)
        try:
            while True:
                src = self.draw_source()
                reference = self.draw_reference()
                self.source_scale = compute_display_scale(src.shape[:2], 1200, 900)
                self.reference_scale = compute_display_scale(reference.shape[:2], 1200, 900)
                source_view = cv2.resize(src, None, fx=self.source_scale, fy=self.source_scale, interpolation=cv2.INTER_AREA)
                reference_view = cv2.resize(reference, None, fx=self.reference_scale, fy=self.reference_scale, interpolation=cv2.INTER_AREA)
                cv2.setWindowTitle(SOURCE_WINDOW, f"{SOURCE_WINDOW} - {self.camera_id}")
                cv2.setWindowTitle(REFERENCE_WINDOW, f"{REFERENCE_WINDOW} - {self.target_camera_id}")
                cv2.imshow(SOURCE_WINDOW, source_view)
                cv2.imshow(REFERENCE_WINDOW, reference_view)

                if cv2.getWindowProperty(SOURCE_WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                    self.save()
                    break
                if cv2.getWindowProperty(REFERENCE_WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                    self.save()
                    break

                key = cv2.waitKey(30) & 0xFF
                if self.pending_exit_confirm and self.handle_exit_confirmation(key):
                    break
                if key in (ord("u"), ord("U")):
                    self.undo()
                elif key in (ord("c"), ord("C")):
                    self.clear()
                elif key in (ord("n"), ord("N"), 13):
                    self.save_and_finish_current()
                elif key in (ord("s"), ord("S")):
                    self.save()
                elif key in (ord("q"), ord("Q"), 27):
                    self.pending_exit_confirm = True
        except KeyboardInterrupt:
            self.pending_exit_confirm = True
            self.save()
        finally:
            cv2.destroyAllWindows()


def main() -> None:
    args = parse_args()
    config, root = load_config(args.config)
    PointAnnotationSession(config, root).run()


if __name__ == "__main__":
    main()
