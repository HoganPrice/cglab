import math

import taichi as ti


ti.init(arch=ti.gpu)

WIDTH, HEIGHT = 960, 540
EPS = 1e-4
INF = 1e8
MAX_UI_BOUNCES = 5
FOV_SCALE = math.tan(0.5 * math.radians(45.0))

pixels = ti.Vector.field(3, dtype=ti.f32, shape=(WIDTH, HEIGHT))


@ti.func
def reflect(direction, normal):
    return direction - 2.0 * direction.dot(normal) * normal


@ti.func
def intersect_sphere(ray_origin, ray_dir, center, radius):
    hit = 0
    t = INF
    normal = ti.Vector([0.0, 1.0, 0.0])
    oc = ray_origin - center
    b = oc.dot(ray_dir)
    c = oc.dot(oc) - radius * radius
    disc = b * b - c
    if disc > 0.0:
        s = ti.sqrt(disc)
        t0 = -b - s
        t1 = -b + s
        candidate = t0
        if candidate < EPS:
            candidate = t1
        if candidate > EPS:
            hit = 1
            t = candidate
            p = ray_origin + t * ray_dir
            normal = (p - center).normalized()
    return hit, t, normal


@ti.func
def intersect_scene(ray_origin, ray_dir):
    hit_any = 0
    best_t = INF
    best_normal = ti.Vector([0.0, 1.0, 0.0])
    material = 0

    if ti.abs(ray_dir.y) > 1e-6:
        t_plane = (-1.0 - ray_origin.y) / ray_dir.y
        if t_plane > EPS and t_plane < best_t:
            hit_any = 1
            best_t = t_plane
            best_normal = ti.Vector([0.0, 1.0, 0.0])
            material = 1

    hit, t, normal = intersect_sphere(
        ray_origin, ray_dir, ti.Vector([-1.5, 0.0, 0.0]), 1.0
    )
    if hit == 1 and t < best_t:
        hit_any = 1
        best_t = t
        best_normal = normal
        material = 2

    hit, t, normal = intersect_sphere(
        ray_origin, ray_dir, ti.Vector([1.5, 0.0, 0.0]), 1.0
    )
    if hit == 1 and t < best_t:
        hit_any = 1
        best_t = t
        best_normal = normal
        material = 3

    return hit_any, best_t, best_normal, material


@ti.func
def material_color(material, p):
    color = ti.Vector([1.0, 1.0, 1.0])
    if material == 1:
        checker = (ti.floor(p.x) + ti.floor(p.z)) % 2
        if checker == 0:
            color = ti.Vector([0.92, 0.92, 0.92])
        else:
            color = ti.Vector([0.08, 0.08, 0.08])
    elif material == 2:
        color = ti.Vector([0.95, 0.12, 0.08])
    elif material == 3:
        color = ti.Vector([0.82, 0.84, 0.86])
    return color


@ti.func
def shade_diffuse(p, normal, view_dir, material, light_pos):
    base = material_color(material, p)
    light_vec = light_pos - p
    light_dist = light_vec.norm()
    light_dir = light_vec / light_dist

    shadow_origin = p + normal * EPS
    shadow_hit, shadow_t, _, _ = intersect_scene(shadow_origin, light_dir)
    in_shadow = shadow_hit == 1 and shadow_t < light_dist

    ambient = 0.12 * base
    color = ambient
    if not in_shadow:
        diff = ti.max(normal.dot(light_dir), 0.0)
        half_dir = (light_dir + view_dir).normalized()
        spec = ti.pow(ti.max(normal.dot(half_dir), 0.0), 64.0)
        color = ambient + base * diff * 0.86 + ti.Vector([1.0, 1.0, 1.0]) * spec * 0.25
    return color


@ti.func
def sky_color(ray_dir):
    t = 0.5 * (ray_dir.y + 1.0)
    return (1.0 - t) * ti.Vector([0.78, 0.84, 0.92]) + t * ti.Vector([0.45, 0.62, 0.90])


@ti.kernel
def render(light_x: ti.f32, light_y: ti.f32, light_z: ti.f32, max_bounces: ti.i32):
    camera_pos = ti.Vector([0.0, 1.1, 6.0])
    look_at = ti.Vector([0.0, 0.0, 0.0])
    forward = (look_at - camera_pos).normalized()
    world_up = ti.Vector([0.0, 1.0, 0.0])
    right = forward.cross(world_up).normalized()
    up = right.cross(forward).normalized()
    aspect = WIDTH / HEIGHT
    scale = FOV_SCALE
    light_pos = ti.Vector([light_x, light_y, light_z])

    for i, j in pixels:
        u = (2.0 * ((i + 0.5) / WIDTH) - 1.0) * aspect * scale
        v = (2.0 * ((j + 0.5) / HEIGHT) - 1.0) * scale
        ray_origin = camera_pos
        ray_dir = (forward + u * right + v * up).normalized()

        final_color = ti.Vector([0.0, 0.0, 0.0])
        throughput = ti.Vector([1.0, 1.0, 1.0])
        done = 0

        for bounce in range(MAX_UI_BOUNCES):
            if bounce < max_bounces and done == 0:
                hit, t, normal, material = intersect_scene(ray_origin, ray_dir)
                if hit == 0:
                    final_color += throughput * sky_color(ray_dir)
                    done = 1
                else:
                    p = ray_origin + t * ray_dir
                    if material == 3:
                        throughput *= ti.Vector([0.8, 0.8, 0.8])
                        ray_origin = p + normal * EPS
                        ray_dir = reflect(ray_dir, normal).normalized()
                    else:
                        view_dir = (-ray_dir).normalized()
                        final_color += throughput * shade_diffuse(
                            p, normal, view_dir, material, light_pos
                        )
                        done = 1

        pixels[i, j] = ti.sqrt(ti.min(final_color, ti.Vector([1.0, 1.0, 1.0])))


def main():
    window = ti.ui.Window("Lab 5 Required - Iterative Ray Tracing", (WIDTH, HEIGHT), vsync=True)
    canvas = window.get_canvas()
    gui = window.get_gui()

    light_x, light_y, light_z = 2.5, 5.0, 3.0
    max_bounces = 3

    while window.running:
        with gui.sub_window("Controls", 0.02, 0.02, 0.25, 0.25):
            light_x = gui.slider_float("Light X", light_x, -6.0, 6.0)
            light_y = gui.slider_float("Light Y", light_y, 0.5, 8.0)
            light_z = gui.slider_float("Light Z", light_z, -6.0, 6.0)
            max_bounces = gui.slider_int("Max Bounces", max_bounces, 1, 5)

        render(light_x, light_y, light_z, max_bounces)
        canvas.set_image(pixels)
        window.show()


if __name__ == "__main__":
    main()
