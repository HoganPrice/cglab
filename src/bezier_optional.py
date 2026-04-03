import numpy as np
import taichi as ti


ti.init(arch=ti.gpu)

WIDTH = 800
HEIGHT = 800
TARGET_SAMPLES = 2000
MAX_CONTROL_POINTS = 100
MAX_CURVE_POINTS = TARGET_SAMPLES + 1
LINE_SAMPLES = 400

BACKGROUND = ti.Vector([0.05, 0.05, 0.05])
CURVE_COLOR = ti.Vector([0.1, 0.9, 0.2])
POLY_COLOR = ti.Vector([0.6, 0.6, 0.6])

pixels = ti.Vector.field(3, dtype=ti.f32, shape=(WIDTH, HEIGHT))
curve_points_field = ti.Vector.field(2, dtype=ti.f32, shape=MAX_CURVE_POINTS)
gui_points = ti.Vector.field(2, dtype=ti.f32, shape=MAX_CONTROL_POINTS)
aa_intensity = ti.field(dtype=ti.f32, shape=(WIDTH, HEIGHT))
curve_count = ti.field(dtype=ti.i32, shape=())
control_count = ti.field(dtype=ti.i32, shape=())


def de_casteljau(points, t):
    if not points:
        return np.array([0.0, 0.0], dtype=np.float32)

    temp = np.array(points, dtype=np.float32)
    n = len(temp)
    for r in range(1, n):
        temp[: n - r] = (1.0 - t) * temp[: n - r] + t * temp[1 : n - r + 1]
    return temp[0]


def sample_bezier(points, n_samples):
    out = np.zeros((n_samples + 1, 2), dtype=np.float32)
    for i in range(n_samples + 1):
        t = i / n_samples
        out[i] = de_casteljau(points, t)
    return out


def sample_uniform_cubic_bspline(points, n_samples):
    n = len(points)
    if n < 4:
        return np.zeros((0, 2), dtype=np.float32)

    pts = np.array(points, dtype=np.float32)
    segments = n - 3
    out = np.zeros((n_samples + 1, 2), dtype=np.float32)

    for k in range(n_samples + 1):
        s = (k / n_samples) * segments
        seg = min(int(s), segments - 1)
        u = s - seg

        p0 = pts[seg]
        p1 = pts[seg + 1]
        p2 = pts[seg + 2]
        p3 = pts[seg + 3]

        u2 = u * u
        u3 = u2 * u

        b0 = (-u3 + 3.0 * u2 - 3.0 * u + 1.0) / 6.0
        b1 = (3.0 * u3 - 6.0 * u2 + 4.0) / 6.0
        b2 = (-3.0 * u3 + 3.0 * u2 + 3.0 * u + 1.0) / 6.0
        b3 = u3 / 6.0

        out[k] = b0 * p0 + b1 * p1 + b2 * p2 + b3 * p3

    return out


@ti.kernel
def clear_pixels():
    for x, y in pixels:
        pixels[x, y] = BACKGROUND


@ti.kernel
def clear_aa():
    for x, y in aa_intensity:
        aa_intensity[x, y] = 0.0


@ti.kernel
def draw_curve_kernel():
    for i in range(curve_count[None]):
        p = curve_points_field[i]
        x = int(p[0] * (WIDTH - 1))
        y = int(p[1] * (HEIGHT - 1))
        if 0 <= x < WIDTH and 0 <= y < HEIGHT:
            pixels[x, y] = CURVE_COLOR


@ti.kernel
def draw_curve_aa_kernel():
    for i in range(curve_count[None]):
        p = curve_points_field[i]
        fx = p[0] * (WIDTH - 1)
        fy = p[1] * (HEIGHT - 1)
        cx = int(fx)
        cy = int(fy)

        for dx, dy in ti.ndrange((-1, 2), (-1, 2)):
            x = cx + dx
            y = cy + dy
            if 0 <= x < WIDTH and 0 <= y < HEIGHT:
                dist = ti.sqrt((fx - x) * (fx - x) + (fy - y) * (fy - y))
                w = ti.max(0.0, 1.0 - dist / 1.5)
                ti.atomic_max(aa_intensity[x, y], w)


@ti.kernel
def compose_aa_to_pixels():
    for x, y in pixels:
        w = aa_intensity[x, y]
        pixels[x, y] = BACKGROUND * (1.0 - w) + CURVE_COLOR * w


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
    window = ti.ui.Window("Bezier/B-spline - Optional", (WIDTH, HEIGHT))
    canvas = window.get_canvas()

    control_points = []
    curve_np = np.zeros((MAX_CURVE_POINTS, 2), dtype=np.float32)

    mode = "bezier"
    aa_enabled = True

    while window.running:
        while window.get_event(ti.ui.PRESS):
            if window.event.key == ti.ui.LMB and len(control_points) < MAX_CONTROL_POINTS:
                pos = window.get_cursor_pos()
                control_points.append([pos[0], pos[1]])
            elif window.event.key == "c":
                control_points.clear()
            elif window.event.key == "b":
                mode = "bspline" if mode == "bezier" else "bezier"
            elif window.event.key == "a":
                aa_enabled = not aa_enabled

        clear_pixels()

        gui_points_np = np.full((MAX_CONTROL_POINTS, 2), -10.0, dtype=np.float32)
        n_ctrl = len(control_points)
        if n_ctrl > 0:
            gui_points_np[:n_ctrl] = np.array(control_points, dtype=np.float32)
        gui_points.from_numpy(gui_points_np)

        n_curve = 0
        if mode == "bezier" and n_ctrl >= 2:
            sampled = sample_bezier(control_points, TARGET_SAMPLES)
            n_curve = sampled.shape[0]
            curve_np[:n_curve] = sampled
        elif mode == "bspline" and n_ctrl >= 4:
            sampled = sample_uniform_cubic_bspline(control_points, TARGET_SAMPLES)
            n_curve = sampled.shape[0]
            if n_curve > 0:
                curve_np[:n_curve] = sampled

        if n_curve > 0:
            curve_points_field.from_numpy(curve_np)
            curve_count[None] = n_curve
            if aa_enabled:
                clear_aa()
                draw_curve_aa_kernel()
                compose_aa_to_pixels()
            else:
                draw_curve_kernel()

        if n_ctrl >= 2:
            control_count[None] = n_ctrl
            draw_control_polygon_kernel()

        canvas.set_image(pixels)
        canvas.circles(gui_points, radius=0.008, color=(1.0, 0.2, 0.2))
        window.show()


if __name__ == "__main__":
    main()
