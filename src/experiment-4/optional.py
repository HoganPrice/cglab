import taichi as ti


ti.init(arch=ti.gpu)

WIDTH = 960
HEIGHT = 640

pixels = ti.Vector.field(3, dtype=ti.f32, shape=(WIDTH, HEIGHT))

camera_pos = ti.Vector([0.0, 0.0, 5.0])
DEFAULT_LIGHT_POS = (2.0, 3.0, 4.0)
light_color = ti.Vector([1.0, 1.0, 1.0])
background_color = ti.Vector([0.02, 0.13, 0.16])

sphere_center = ti.Vector([-1.2, -0.2, 0.0])
sphere_radius = 1.2
sphere_color = ti.Vector([0.8, 0.1, 0.1])

cone_apex = ti.Vector([1.2, 1.2, 0.0])
cone_base_y = -1.4
cone_radius = 1.2
cone_height = 2.6
cone_color = ti.Vector([0.6, 0.2, 0.8])

INF = 1.0e8
EPS = 1.0e-4


@ti.func
def reflect(incident, normal):
    return incident - 2.0 * incident.dot(normal) * normal


@ti.func
def hit_sphere(ro, rd):
    t = INF
    normal = ti.Vector([0.0, 0.0, 0.0])

    oc = ro - sphere_center
    a = rd.dot(rd)
    b = 2.0 * oc.dot(rd)
    c = oc.dot(oc) - sphere_radius * sphere_radius
    disc = b * b - 4.0 * a * c

    if disc >= 0.0:
        sqrtd = ti.sqrt(disc)
        t0 = (-b - sqrtd) / (2.0 * a)
        t1 = (-b + sqrtd) / (2.0 * a)
        if t0 > EPS:
            t = t0
        elif t1 > EPS:
            t = t1

        if t < INF:
            p = ro + rd * t
            normal = (p - sphere_center).normalized()

    return t, normal


@ti.func
def hit_cone(ro, rd):
    t = INF
    normal = ti.Vector([0.0, 0.0, 0.0])
    k = cone_radius / cone_height
    k2 = k * k

    co = ro - cone_apex
    a = rd.x * rd.x + rd.z * rd.z - k2 * rd.y * rd.y
    b = 2.0 * (co.x * rd.x + co.z * rd.z - k2 * co.y * rd.y)
    c = co.x * co.x + co.z * co.z - k2 * co.y * co.y
    disc = b * b - 4.0 * a * c

    if ti.abs(a) > 1.0e-6 and disc >= 0.0:
        sqrtd = ti.sqrt(disc)
        t0 = (-b - sqrtd) / (2.0 * a)
        t1 = (-b + sqrtd) / (2.0 * a)

        if t0 > EPS:
            p0 = ro + rd * t0
            if p0.y >= cone_base_y and p0.y <= cone_apex.y and t0 < t:
                t = t0
        if t1 > EPS:
            p1 = ro + rd * t1
            if p1.y >= cone_base_y and p1.y <= cone_apex.y and t1 < t:
                t = t1

        if t < INF:
            p = ro + rd * t
            q = p - cone_apex
            normal = ti.Vector([q.x, -k2 * q.y, q.z]).normalized()

    if ti.abs(rd.y) > 1.0e-6:
        t_cap = (cone_base_y - ro.y) / rd.y
        if t_cap > EPS and t_cap < t:
            p_cap = ro + rd * t_cap
            dx = p_cap.x - cone_apex.x
            dz = p_cap.z - cone_apex.z
            if dx * dx + dz * dz <= cone_radius * cone_radius:
                t = t_cap
                normal = ti.Vector([0.0, -1.0, 0.0])

    return t, normal


@ti.func
def nearest_hit(ro, rd):
    t_min = INF
    normal = ti.Vector([0.0, 0.0, 0.0])
    base_color = ti.Vector([0.0, 0.0, 0.0])
    hit = 0

    t_sphere, n_sphere = hit_sphere(ro, rd)
    if t_sphere < t_min:
        t_min = t_sphere
        normal = n_sphere
        base_color = sphere_color
        hit = 1

    t_cone, n_cone = hit_cone(ro, rd)
    if t_cone < t_min:
        t_min = t_cone
        normal = n_cone
        base_color = cone_color
        hit = 1

    return hit, t_min, normal, base_color


@ti.func
def in_shadow(point, light_dir, light_distance):
    shadow = 0
    shadow_ro = point + light_dir * EPS * 10.0

    t_sphere, n_sphere_unused = hit_sphere(shadow_ro, light_dir)
    if t_sphere > EPS and t_sphere < light_distance:
        shadow = 1

    t_cone, n_cone_unused = hit_cone(shadow_ro, light_dir)
    if t_cone > EPS and t_cone < light_distance:
        shadow = 1

    return shadow


@ti.func
def shade(
    point,
    normal,
    base_color,
    ka: ti.f32,
    kd: ti.f32,
    ks: ti.f32,
    shininess: ti.f32,
    light_position,
):
    n = normal.normalized()
    to_light = light_position - point
    light_distance = to_light.norm()
    l = to_light / light_distance
    v = (camera_pos - point).normalized()
    h = (l + v).normalized()

    ambient = ka * light_color * base_color
    color = ambient

    if in_shadow(point, l, light_distance) == 0:
        diff_factor = ti.max(0.0, n.dot(l))
        diffuse = kd * diff_factor * light_color * base_color

        # Optional task: Blinn-Phong uses the half-vector H instead of the
        # reflection vector R. It usually gives broader, smoother highlights.
        spec_factor = ti.pow(ti.max(0.0, n.dot(h)), shininess)
        specular = ks * spec_factor * light_color

        color = ambient + diffuse + specular

    return ti.math.clamp(color, 0.0, 1.0)


@ti.kernel
def render(
    ka: ti.f32,
    kd: ti.f32,
    ks: ti.f32,
    shininess: ti.f32,
    light_x: ti.f32,
    light_y: ti.f32,
    light_z: ti.f32,
):
    light_position = ti.Vector([light_x, light_y, light_z])

    for i, j in pixels:
        aspect = ti.cast(WIDTH, ti.f32) / ti.cast(HEIGHT, ti.f32)
        u = (2.0 * (ti.cast(i, ti.f32) + 0.5) / ti.cast(WIDTH, ti.f32) - 1.0) * aspect
        v = 2.0 * (ti.cast(j, ti.f32) + 0.5) / ti.cast(HEIGHT, ti.f32) - 1.0

        ro = camera_pos
        rd = ti.Vector([u * 2.2, v * 2.2, -5.0]).normalized()

        hit, t, normal, base_color = nearest_hit(ro, rd)
        color = background_color
        if hit == 1:
            point = ro + rd * t
            color = shade(
                point, normal, base_color, ka, kd, ks, shininess, light_position
            )

        pixels[i, j] = color


def main():
    ka = 0.2
    kd = 0.7
    ks = 0.5
    shininess = 32.0
    light_x, light_y, light_z = DEFAULT_LIGHT_POS

    window = ti.ui.Window(
        "CG Lab 4 - Optional Blinn-Phong + Hard Shadow", (WIDTH, HEIGHT)
    )
    canvas = window.get_canvas()
    gui = window.get_gui()

    while window.running:
        with gui.sub_window("Material", 0.02, 0.02, 0.28, 0.26):
            ka = gui.slider_float("Ka", ka, 0.0, 1.0)
            kd = gui.slider_float("Kd", kd, 0.0, 1.0)
            ks = gui.slider_float("Ks", ks, 0.0, 1.0)
            shininess = gui.slider_float("Shininess", shininess, 1.0, 128.0)

        with gui.sub_window("Light Position", 0.02, 0.30, 0.28, 0.22):
            light_x = gui.slider_float("X", light_x, -5.0, 5.0)
            light_y = gui.slider_float("Y", light_y, -5.0, 6.0)
            light_z = gui.slider_float("Z", light_z, -2.0, 8.0)

        render(ka, kd, ks, shininess, light_x, light_y, light_z)
        canvas.set_image(pixels)
        window.show()


if __name__ == "__main__":
    main()
