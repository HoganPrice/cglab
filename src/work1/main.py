import argparse
import math
from typing import Iterable

import numpy as np
import taichi as ti

from .config import (
    ASPECT_RATIO,
    BACKGROUND_COLOR,
    EDGE_COLORS,
    EYE_FOV,
    EYE_POS,
    LINE_RADIUS,
    ROTATE_STEP_DEG,
    TRIANGLE_EDGES,
    TRIANGLE_VERTICES,
    WINDOW_RES,
    WINDOW_TITLE,
    Z_FAR,
    Z_NEAR,
)


ti.init(arch=ti.gpu)


def get_model_matrix(angle: float) -> np.ndarray:
    angle_rad = angle * math.pi / 180.0
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)

    return np.array(
        [
            [cos_a, -sin_a, 0.0, 0.0],
            [sin_a, cos_a, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )


def get_view_matrix(eye_pos: Iterable[float]) -> np.ndarray:
    eye_x, eye_y, eye_z = eye_pos

    return np.array(
        [
            [1.0, 0.0, 0.0, -eye_x],
            [0.0, 1.0, 0.0, -eye_y],
            [0.0, 0.0, 1.0, -eye_z],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )


def get_projection_matrix(
    eye_fov: float,
    aspect_ratio: float,
    zNear: float,
    zFar: float,
) -> np.ndarray:
    fov_rad = eye_fov * math.pi / 180.0

    n = -zNear
    f = -zFar

    t = math.tan(fov_rad / 2.0) * abs(n)
    b = -t
    r = aspect_ratio * t
    l = -r

    persp_to_ortho = np.array(
        [
            [n, 0.0, 0.0, 0.0],
            [0.0, n, 0.0, 0.0],
            [0.0, 0.0, n + f, -n * f],
            [0.0, 0.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )

    ortho_translate = np.array(
        [
            [1.0, 0.0, 0.0, -(r + l) / 2.0],
            [0.0, 1.0, 0.0, -(t + b) / 2.0],
            [0.0, 0.0, 1.0, -(n + f) / 2.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )

    ortho_scale = np.array(
        [
            [2.0 / (r - l), 0.0, 0.0, 0.0],
            [0.0, 2.0 / (t - b), 0.0, 0.0],
            [0.0, 0.0, 2.0 / (n - f), 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )

    ortho = ortho_scale @ ortho_translate
    return ortho @ persp_to_ortho


def project_vertices(mvp: np.ndarray) -> np.ndarray:
    screen_points = []

    for vertex in TRIANGLE_VERTICES:
        point_homo = np.array([vertex[0], vertex[1], vertex[2], 1.0], dtype=np.float32)
        clip = mvp @ point_homo
        ndc = clip[:3] / clip[3]

        x = (ndc[0] + 1.0) * 0.5
        y = (ndc[1] + 1.0) * 0.5
        screen_points.append([x, y])

    return np.array(screen_points, dtype=np.float32)


def run() -> None:
    gui = ti.GUI(WINDOW_TITLE, res=WINDOW_RES)

    angle = 0.0
    view = get_view_matrix(EYE_POS)
    projection = get_projection_matrix(EYE_FOV, ASPECT_RATIO, Z_NEAR, Z_FAR)

    while gui.running:
        for event in gui.get_events(ti.GUI.PRESS):
            if event.key == ti.GUI.ESCAPE:
                gui.running = False
            elif event.key == "a":
                angle += ROTATE_STEP_DEG
            elif event.key == "d":
                angle -= ROTATE_STEP_DEG

        model = get_model_matrix(angle)
        mvp = projection @ view @ model
        points_2d = project_vertices(mvp)

        gui.clear(BACKGROUND_COLOR)

        for edge_idx, (start_idx, end_idx) in enumerate(TRIANGLE_EDGES):
            gui.line(
                begin=points_2d[start_idx],
                end=points_2d[end_idx],
                radius=LINE_RADIUS,
                color=EDGE_COLORS[edge_idx % len(EDGE_COLORS)],
            )

        gui.text(content="A/D: Rotate | ESC: Exit", pos=(0.02, 0.96), color=0xFFFFFF)
        gui.show()


def dry_run() -> None:
    model = get_model_matrix(30.0)
    view = get_view_matrix(EYE_POS)
    projection = get_projection_matrix(EYE_FOV, ASPECT_RATIO, Z_NEAR, Z_FAR)
    mvp = projection @ view @ model
    points_2d = project_vertices(mvp)

    print("Model matrix:\n", model)
    print("View matrix:\n", view)
    print("Projection matrix:\n", projection)
    print("Projected 2D points:\n", points_2d)


def main() -> None:
    parser = argparse.ArgumentParser(description="Experiment 2 MVP demo")
    parser.add_argument("--dry-run", action="store_true", help="print matrices and projected points")
    args = parser.parse_args()

    if args.dry_run:
        dry_run()
        return

    run()


if __name__ == "__main__":
    main()
