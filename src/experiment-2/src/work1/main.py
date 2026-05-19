import argparse
import math
from typing import Iterable

import numpy as np
import taichi as ti

from .config import (
    ASPECT_RATIO,
    BACKGROUND_COLOR,
    CUBE_EDGES,
    CUBE_VERTICES,
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


def get_model_matrix(angle_z: float, angle_y: float = 0.0, angle_x: float = 0.0) -> np.ndarray:
    angle_x_rad = angle_x * math.pi / 180.0
    angle_y_rad = angle_y * math.pi / 180.0
    angle_z_rad = angle_z * math.pi / 180.0

    cos_x = math.cos(angle_x_rad)
    sin_x = math.sin(angle_x_rad)
    cos_y = math.cos(angle_y_rad)
    sin_y = math.sin(angle_y_rad)
    cos_z = math.cos(angle_z_rad)
    sin_z = math.sin(angle_z_rad)

    rotation_x = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, cos_x, -sin_x, 0.0],
            [0.0, sin_x, cos_x, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )

    rotation_y = np.array(
        [
            [cos_y, 0.0, sin_y, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [-sin_y, 0.0, cos_y, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )

    rotation_z = np.array(
        [
            [cos_z, -sin_z, 0.0, 0.0],
            [sin_z, cos_z, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )

    return rotation_z @ rotation_y @ rotation_x


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


def project_vertices(vertices: Iterable[Iterable[float]], mvp: np.ndarray) -> np.ndarray:
    screen_points = []

    for vertex in vertices:
        point_homo = np.array([vertex[0], vertex[1], vertex[2], 1.0], dtype=np.float32)
        clip = mvp @ point_homo
        ndc = clip[:3] / clip[3]

        x = (ndc[0] + 1.0) * 0.5
        y = (ndc[1] + 1.0) * 0.5
        screen_points.append([x, y])

    return np.array(screen_points, dtype=np.float32)


def run() -> None:
    gui = ti.GUI(WINDOW_TITLE, res=WINDOW_RES)

    angle_z = 0.0
    angle_y = 0.0
    angle_x = 0.0
    use_cube = True

    view = get_view_matrix(EYE_POS)
    projection = get_projection_matrix(EYE_FOV, ASPECT_RATIO, Z_NEAR, Z_FAR)

    while gui.running:
        for event in gui.get_events(ti.GUI.PRESS):
            if event.key == ti.GUI.ESCAPE:
                gui.running = False
            elif event.key == "a":
                angle_z += ROTATE_STEP_DEG
            elif event.key == "d":
                angle_z -= ROTATE_STEP_DEG
            elif event.key == "w":
                angle_y += ROTATE_STEP_DEG
            elif event.key == "s":
                angle_y -= ROTATE_STEP_DEG
            elif event.key == "q":
                angle_x += ROTATE_STEP_DEG
            elif event.key == "e":
                angle_x -= ROTATE_STEP_DEG
            elif event.key == "t":
                use_cube = not use_cube

        model = get_model_matrix(angle_z, angle_y, angle_x)
        mvp = projection @ view @ model
        active_vertices = CUBE_VERTICES if use_cube else TRIANGLE_VERTICES
        active_edges = CUBE_EDGES if use_cube else TRIANGLE_EDGES
        points_2d = project_vertices(active_vertices, mvp)

        gui.clear(BACKGROUND_COLOR)

        for edge_idx, (start_idx, end_idx) in enumerate(active_edges):
            gui.line(
                begin=points_2d[start_idx],
                end=points_2d[end_idx],
                radius=LINE_RADIUS,
                color=EDGE_COLORS[edge_idx % len(EDGE_COLORS)],
            )

        shape_name = "Cube" if use_cube else "Triangle"
        gui.text(
            content=f"Shape: {shape_name} | T: Toggle",
            pos=(0.02, 0.96),
            color=0xFFFFFF,
        )
        gui.text(
            content="A/D: Z  W/S: Y  Q/E: X  ESC: Exit",
            pos=(0.02, 0.92),
            color=0xFFFFFF,
        )
        gui.show()


def dry_run() -> None:
    model = get_model_matrix(angle_z=30.0, angle_y=20.0)
    view = get_view_matrix(EYE_POS)
    projection = get_projection_matrix(EYE_FOV, ASPECT_RATIO, Z_NEAR, Z_FAR)
    mvp = projection @ view @ model
    triangle_points_2d = project_vertices(TRIANGLE_VERTICES, mvp)
    cube_points_2d = project_vertices(CUBE_VERTICES, mvp)

    print("Model matrix:\n", model)
    print("View matrix:\n", view)
    print("Projection matrix:\n", projection)
    print("Projected 2D points (triangle):\n", triangle_points_2d)
    print("Projected 2D points (cube):\n", cube_points_2d)


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
