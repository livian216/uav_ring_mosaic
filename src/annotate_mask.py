from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np

from utils import (
    compute_display_scale,
    get_camera_ids,
    get_input_paths,
    get_mask_paths,
    info,
    load_config,
    load_image,
    parse_args,
    save_image,
)


WINDOW_NAME = "Building Mask Annotation"


class MaskAnnotationSession:
    def __init__(self, config: dict, root: Path) -> None:
        self.config = config
        self.root = root
        self.camera_ids = get_camera_ids(config, mode="image")
        self.image_paths = get_input_paths(config, root, mode="image")
        self.mask_paths = get_mask_paths(config, root)
        self.index = 0
        self.points: List[List[int]] = []
        self.display_scale = 1.0
        self.dragging_index = None
        self.polygon_closed = False
        self.pending_exit_confirm = False

    @property
    def camera_id(self) -> str:
        return self.camera_ids[self.index]

    def current_image(self) -> np.ndarray:
        return load_image(self.image_paths[self.camera_id])

    def current_mask_path(self) -> Path:
        return self.mask_paths[self.camera_id]

    def on_click(self, event, x, y, flags, param) -> None:
        _ = flags, param
        mapped = [int(round(x / self.display_scale)), int(round(y / self.display_scale))]
        if event == cv2.EVENT_LBUTTONDOWN:
            nearby = self.find_nearby_vertex(x, y)
            if nearby is not None:
                self.dragging_index = nearby
            else:
                self.points.append(mapped)
                self.polygon_closed = False
        elif event == cv2.EVENT_RBUTTONDOWN:
            self.polygon_closed = len(self.points) >= 3
        elif event == cv2.EVENT_MOUSEMOVE and self.dragging_index is not None:
            self.points[self.dragging_index] = mapped
        elif event == cv2.EVENT_LBUTTONUP:
            self.dragging_index = None

    def find_nearby_vertex(self, x: int, y: int, radius: int = 18):
        for idx, point in enumerate(self.points):
            px = point[0] * self.display_scale
            py = point[1] * self.display_scale
            if (px - x) ** 2 + (py - y) ** 2 <= radius ** 2:
                return idx
        return None

    def draw(self) -> np.ndarray:
        image = self.current_image().copy()
        for idx, point in enumerate(self.points):
            cv2.circle(image, tuple(point), 5, (0, 255, 0), -1)
            cv2.putText(image, str(idx + 1), tuple(point), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        if len(self.points) >= 2:
            cv2.polylines(
                image,
                [np.asarray(self.points, dtype=np.int32)],
                self.polygon_closed,
                (0, 255, 255),
                2,
            )
        cv2.putText(
            image,
            "Left: add/drag vertex | Right/F: close polygon | S: save | N: next | Q/Esc: exit confirm",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
        )
        cv2.putText(
            image,
            "U: undo last vertex | R: reset | Closed polygon remains editable by dragging vertices",
            (20, 58),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
        )
        if self.pending_exit_confirm:
            cv2.putText(
                image,
                "Exit? Press Y to save and exit, N to exit without saving, any other key to cancel.",
                (20, 86),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 165, 255),
                2,
            )
        return image

    def build_mask(self) -> np.ndarray:
        image = self.current_image()
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        if len(self.points) >= 3:
            polygon = np.asarray(self.points, dtype=np.int32)
            cv2.fillPoly(mask, [polygon], 255)
        return mask

    def save_current_mask(self) -> None:
        if len(self.points) < 3:
            info(f"{self.camera_id} requires at least 3 vertices to save a polygon.")
            return
        self.polygon_closed = True
        mask = self.build_mask()
        save_image(self.current_mask_path(), mask)
        info(f"Saved mask to {self.current_mask_path()}")

    def reset(self) -> None:
        self.points = []
        self.polygon_closed = False

    def undo(self) -> None:
        if self.points:
            self.points.pop()
        if len(self.points) < 3:
            self.polygon_closed = False

    def next_camera(self) -> None:
        self.reset()
        self.index = (self.index + 1) % len(self.camera_ids)

    def handle_exit_confirmation(self, key: int) -> bool:
        if not self.pending_exit_confirm:
            return False
        if key in (ord("y"), ord("Y")):
            if len(self.points) >= 3:
                self.save_current_mask()
            return True
        if key in (ord("n"), ord("N")):
            return True
        self.pending_exit_confirm = False
        return False

    def run(self) -> None:
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(WINDOW_NAME, self.on_click)
        try:
            while True:
                image = self.draw()
                self.display_scale = compute_display_scale(image.shape[:2], 1200, 900)
                view = cv2.resize(image, None, fx=self.display_scale, fy=self.display_scale, interpolation=cv2.INTER_AREA)
                cv2.setWindowTitle(WINDOW_NAME, f"{WINDOW_NAME} - {self.camera_id}")
                cv2.imshow(WINDOW_NAME, view)
                if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                    if len(self.points) >= 3:
                        self.save_current_mask()
                    break
                key = cv2.waitKey(30) & 0xFF
                if self.pending_exit_confirm and self.handle_exit_confirmation(key):
                    break
                if key in (ord("u"), ord("U")):
                    self.undo()
                elif key in (ord("r"), ord("R")):
                    self.reset()
                elif key in (ord("f"), ord("F")):
                    self.polygon_closed = len(self.points) >= 3
                elif key in (ord("s"), ord("S")):
                    self.save_current_mask()
                elif key in (ord("n"), ord("N")):
                    if len(self.points) >= 3:
                        self.save_current_mask()
                    self.next_camera()
                elif key in (ord("q"), ord("Q"), 27):
                    self.pending_exit_confirm = True
        except KeyboardInterrupt:
            if len(self.points) >= 3:
                self.save_current_mask()
        finally:
            cv2.destroyAllWindows()


def main() -> None:
    args = parse_args()
    config, root = load_config(args.config)
    MaskAnnotationSession(config, root).run()


if __name__ == "__main__":
    main()
