import numpy as np
import taichi as ti


ti.init(arch=ti.gpu)

WIDTH = 800
HEIGHT = 800
NUM_SEGMENTS = 1000
MAX_CONTROL_POINTS = 100
LINE_SAMPLES = 400

BACKGROUND = ti.Vector([0.05, 0.05, 0.05])
CURVE_COLOR = ti.Vector([0.1, 0.9, 0.2])
POLY_COLOR = ti.Vector([0.6, 0.6, 0.6])

pixels = ti.Vector.field(3, dtype=ti.f32, shape=(WIDTH, HEIGHT))
curve_points_field = ti.Vector.field(2, dtype=ti.f32, shape=NUM_SEGMENTS + 1)
gui_points = ti.Vector.field(2, dtype=ti.f32, shape=MAX_CONTROL_POINTS)
curve_count = ti.field(dtype=ti.i32, shape=())
control_count = ti.field(dtype=ti.i32, shape=())


def de_casteljau(points, t):
    """Compute one point on a Bezier curve by recursive linear interpolation."""
    if not points:
        return np.array([0.0, 0.0], dtype=np.float32)

    temp = np.array(points, dtype=np.float32)
    n = len(temp)
    for r in range(1, n):
        temp[: n - r] = (1.0 - t) * temp[: n - r] + t * temp[1 : n - r + 1]
    return temp[0]


@ti.kernel
def clear_pixels():
    for x, y in pixels:
        pixels[x, y] = BACKGROUND


@ti.kernel
def draw_curve_kernel():
    for i in range(curve_count[None]):
        p = curve_points_field[i]
        x = int(p[0] * (WIDTH - 1))
        y = int(p[1] * (HEIGHT - 1))
        if 0 <= x < WIDTH and 0 <= y < HEIGHT:
            pixels[x, y] = CURVE_COLOR


@ti.kernel
def draw_control_polygon_kernel():
    for i in range(control_count[None] - 1):
        p0 = gui_points[i]
        p1 = gui_points[i + 1]
        for s in range(LINE_SAMPLES + 1):
            t = s / LINE_SAMPLES
            p = (1.0 - t) * p0 + t * p1
            x = int(p[0] * (WIDTH - 1))
            y = int(p[1] * (HEIGHT - 1))
            if 0 <= x < WIDTH and 0 <= y < HEIGHT:
                pixels[x, y] = POLY_COLOR


def main():
    window = ti.ui.Window("Bezier Curve - Required", (WIDTH, HEIGHT))
    canvas = window.get_canvas()

    control_points = []
    curve_np = np.zeros((NUM_SEGMENTS + 1, 2), dtype=np.float32)

    while window.running:
        while window.get_event(ti.ui.PRESS):
            if window.event.key == ti.ui.LMB and len(control_points) < MAX_CONTROL_POINTS:
                pos = window.get_cursor_pos()
                control_points.append([pos[0], pos[1]])
            elif window.event.key == "c":
                control_points.clear()

        clear_pixels()

        gui_points_np = np.full((MAX_CONTROL_POINTS, 2), -10.0, dtype=np.float32)
        n_ctrl = len(control_points)
        if n_ctrl > 0:
            gui_points_np[:n_ctrl] = np.array(control_points, dtype=np.float32)
        gui_points.from_numpy(gui_points_np)

        if n_ctrl >= 2:
            for i in range(NUM_SEGMENTS + 1):
                t = i / NUM_SEGMENTS
                curve_np[i] = de_casteljau(control_points, t)
            curve_points_field.from_numpy(curve_np)
            curve_count[None] = NUM_SEGMENTS + 1
            control_count[None] = n_ctrl
            draw_curve_kernel()
            draw_control_polygon_kernel()

        canvas.set_image(pixels)
        canvas.circles(gui_points, radius=0.008, color=(1.0, 0.2, 0.2))
        window.show()


if __name__ == "__main__":
    main()
