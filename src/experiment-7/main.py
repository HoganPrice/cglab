import taichi as ti


ti.init(arch=ti.gpu)

N = 20
NUM_PARTICLES = N * N
STRUCTURAL_SPRINGS = 2 * N * (N - 1)
SHEAR_SPRINGS = 2 * (N - 1) * (N - 1)
BENDING_SPRINGS = 2 * N * (N - 2)
NUM_SPRINGS = STRUCTURAL_SPRINGS + SHEAR_SPRINGS + BENDING_SPRINGS
NUM_TRIANGLES = 2 * (N - 1) * (N - 1)
MAX_IMPLICIT_ITERATIONS = 12

MASS = 1.0
GRAVITY = ti.Vector([0.0, -9.8, 0.0])
CLOTH_WIDTH = 3.2
SPACING = CLOTH_WIDTH / (N - 1)
BALL_CENTER = ti.Vector([0.0, -1.15, 0.0])
BALL_RADIUS = 0.55
DT = 5.0e-4
SPRING_STIFFNESS = 10000.0
MAX_VELOCITY = 50.0
PHYSICS_SUBSTEPS = 20
IMPLICIT_ITERATIONS = 5
USE_SHEAR_SPRINGS = 1
USE_BENDING_SPRINGS = 1
ENABLE_SPHERE_COLLISION = 1

positions = ti.Vector.field(3, dtype=ti.f32, shape=NUM_PARTICLES)
velocities = ti.Vector.field(3, dtype=ti.f32, shape=NUM_PARTICLES)
forces = ti.Vector.field(3, dtype=ti.f32, shape=NUM_PARTICLES)
pred_positions = ti.Vector.field(3, dtype=ti.f32, shape=NUM_PARTICLES)
pred_velocities = ti.Vector.field(3, dtype=ti.f32, shape=NUM_PARTICLES)
pred_forces = ti.Vector.field(3, dtype=ti.f32, shape=NUM_PARTICLES)
fixed = ti.field(dtype=ti.i32, shape=NUM_PARTICLES)

spring_a = ti.field(dtype=ti.i32, shape=NUM_SPRINGS)
spring_b = ti.field(dtype=ti.i32, shape=NUM_SPRINGS)
spring_type = ti.field(dtype=ti.i32, shape=NUM_SPRINGS)
rest_length = ti.field(dtype=ti.f32, shape=NUM_SPRINGS)
spring_count = ti.field(dtype=ti.i32, shape=())

line_indices = ti.field(dtype=ti.i32, shape=NUM_SPRINGS * 2)
triangle_indices = ti.field(dtype=ti.i32, shape=NUM_TRIANGLES * 3)
ball_position = ti.Vector.field(3, dtype=ti.f32, shape=1)


@ti.func
def particle_id(i, j):
    return i * N + j


@ti.func
def add_spring(a, b, kind):
    sid = ti.atomic_add(spring_count[None], 1)
    spring_a[sid] = a
    spring_b[sid] = b
    spring_type[sid] = kind
    rest_length[sid] = (positions[a] - positions[b]).norm()
    line_indices[2 * sid] = a
    line_indices[2 * sid + 1] = b


@ti.func
def spring_enabled(kind, use_shear, use_bending):
    enabled = 0
    if kind == 0:
        enabled = 1
    if kind == 1 and use_shear == 1:
        enabled = 1
    if kind == 2 and use_bending == 1:
        enabled = 1
    return enabled


@ti.func
def clamp_velocity(v, max_speed):
    speed = v.norm()
    out = v
    if speed > max_speed:
        out = v / speed * max_speed
    return out


@ti.func
def collide_with_ball(x, v, enable_collision):
    new_x = x
    new_v = v
    if enable_collision == 1:
        offset = x - BALL_CENTER
        dist = offset.norm()
        if dist < BALL_RADIUS:
            normal = offset.normalized()
            new_x = BALL_CENTER + normal * BALL_RADIUS
            vn = new_v.dot(normal)
            if vn < 0.0:
                new_v -= normal * vn
    return new_x, new_v


@ti.func
def compute_forces_on(sid, stiffness, damping, use_shear, use_bending, use_pred):
    kind = spring_type[sid]
    if spring_enabled(kind, use_shear, use_bending) == 1:
        a = spring_a[sid]
        b = spring_b[sid]
        xa = positions[a]
        xb = positions[b]
        va = velocities[a]
        vb = velocities[b]
        if use_pred == 1:
            xa = pred_positions[a]
            xb = pred_positions[b]
            va = pred_velocities[a]
            vb = pred_velocities[b]

        delta = xa - xb
        length = delta.norm()
        if length > 1.0e-6:
            direction = delta / length
            spring_force = -stiffness * (length - rest_length[sid]) * direction
            relative_velocity = va - vb
            spring_damping = -damping * relative_velocity.dot(direction) * direction
            total = spring_force + spring_damping

            if use_pred == 0:
                ti.atomic_add(forces[a], total)
                ti.atomic_add(forces[b], -total)
            else:
                ti.atomic_add(pred_forces[a], total)
                ti.atomic_add(pred_forces[b], -total)


@ti.kernel
def init_positions():
    for i, j in ti.ndrange(N, N):
        idx = particle_id(i, j)
        x = (j / (N - 1) - 0.5) * CLOTH_WIDTH
        y = 1.15
        z = (0.5 - i / (N - 1)) * CLOTH_WIDTH
        positions[idx] = ti.Vector([x, y, z])
        velocities[idx] = ti.Vector([0.0, 0.0, 0.0])
        forces[idx] = ti.Vector([0.0, 0.0, 0.0])
        pred_positions[idx] = positions[idx]
        pred_velocities[idx] = velocities[idx]
        pred_forces[idx] = forces[idx]
        fixed[idx] = 0
        if i == 0 and (j == 0 or j == N - 1 or j % 3 == 0):
            fixed[idx] = 1


@ti.kernel
def init_springs():
    spring_count[None] = 0
    for i, j in ti.ndrange(N, N):
        idx = particle_id(i, j)
        if j + 1 < N:
            add_spring(idx, particle_id(i, j + 1), 0)
        if i + 1 < N:
            add_spring(idx, particle_id(i + 1, j), 0)
        if i + 1 < N and j + 1 < N:
            add_spring(idx, particle_id(i + 1, j + 1), 1)
        if i + 1 < N and j - 1 >= 0:
            add_spring(idx, particle_id(i + 1, j - 1), 1)
        if j + 2 < N:
            add_spring(idx, particle_id(i, j + 2), 2)
        if i + 2 < N:
            add_spring(idx, particle_id(i + 2, j), 2)


@ti.kernel
def init_render_indices():
    for i, j in ti.ndrange(N - 1, N - 1):
        cell = i * (N - 1) + j
        a = particle_id(i, j)
        b = particle_id(i, j + 1)
        c = particle_id(i + 1, j)
        d = particle_id(i + 1, j + 1)
        base = 6 * cell
        triangle_indices[base] = a
        triangle_indices[base + 1] = c
        triangle_indices[base + 2] = b
        triangle_indices[base + 3] = b
        triangle_indices[base + 4] = c
        triangle_indices[base + 5] = d


@ti.kernel
def step_explicit(dt: ti.f32, stiffness: ti.f32, damping: ti.f32, max_speed: ti.f32,
                  use_shear: ti.i32, use_bending: ti.i32, enable_collision: ti.i32):
    for i in positions:
        forces[i] = GRAVITY * MASS - damping * velocities[i]
    for sid in range(NUM_SPRINGS):
        compute_forces_on(sid, stiffness, damping, use_shear, use_bending, 0)
    for i in positions:
        if fixed[i] == 0:
            old_v = velocities[i]
            acc = forces[i] / MASS
            positions[i] += old_v * dt
            velocities[i] = clamp_velocity(old_v + acc * dt, max_speed)
            positions[i], velocities[i] = collide_with_ball(
                positions[i], velocities[i], enable_collision
            )
        else:
            velocities[i] = ti.Vector([0.0, 0.0, 0.0])


@ti.kernel
def step_semi_implicit(dt: ti.f32, stiffness: ti.f32, damping: ti.f32,
                       max_speed: ti.f32, use_shear: ti.i32, use_bending: ti.i32,
                       enable_collision: ti.i32):
    for i in positions:
        forces[i] = GRAVITY * MASS - damping * velocities[i]
    for sid in range(NUM_SPRINGS):
        compute_forces_on(sid, stiffness, damping, use_shear, use_bending, 0)
    for i in positions:
        if fixed[i] == 0:
            acc = forces[i] / MASS
            velocities[i] = clamp_velocity(velocities[i] + acc * dt, max_speed)
            positions[i] += velocities[i] * dt
            positions[i], velocities[i] = collide_with_ball(
                positions[i], velocities[i], enable_collision
            )
        else:
            velocities[i] = ti.Vector([0.0, 0.0, 0.0])


@ti.kernel
def step_implicit_iter(dt: ti.f32, stiffness: ti.f32, damping: ti.f32,
                       max_speed: ti.f32, use_shear: ti.i32, use_bending: ti.i32,
                       enable_collision: ti.i32, iterations: ti.i32):
    for i in positions:
        pred_velocities[i] = velocities[i]
        pred_positions[i] = positions[i] + velocities[i] * dt

    for iteration in ti.static(range(MAX_IMPLICIT_ITERATIONS)):
        for i in positions:
            if iteration < iterations:
                pred_forces[i] = GRAVITY * MASS - damping * pred_velocities[i]
        for sid in range(NUM_SPRINGS):
            if iteration < iterations:
                compute_forces_on(sid, stiffness, damping, use_shear, use_bending, 1)
        for i in positions:
            if iteration < iterations:
                if fixed[i] == 0:
                    acc = pred_forces[i] / MASS
                    pred_velocities[i] = clamp_velocity(velocities[i] + acc * dt, max_speed)
                    pred_positions[i] = positions[i] + pred_velocities[i] * dt
                else:
                    pred_positions[i] = positions[i]
                    pred_velocities[i] = ti.Vector([0.0, 0.0, 0.0])

    for i in positions:
        if fixed[i] == 0:
            positions[i], velocities[i] = collide_with_ball(
                pred_positions[i], pred_velocities[i], enable_collision
            )
        else:
            velocities[i] = ti.Vector([0.0, 0.0, 0.0])


def reset_simulation():
    init_positions()
    init_springs()
    init_render_indices()
    ball_position[0] = BALL_CENTER


def main():
    reset_simulation()

    window = ti.ui.Window("Experiment 7 - Mass Spring Cloth", (1280, 720), vsync=True)
    canvas = window.get_canvas()
    scene = ti.ui.Scene()
    camera = ti.ui.Camera()
    camera.position(0.0, 1.25, 5.2)
    camera.lookat(0.0, 0.0, 0.0)
    camera.fov(45)

    solver = 1
    damping = 1.0

    solver_names = ["Explicit Euler", "Semi-Implicit Euler", "Implicit Euler"]

    while window.running:
        camera.track_user_inputs(window, movement_speed=0.03, hold_key=ti.ui.RMB)

        gui = window.get_gui()
        with gui.sub_window("Controls", 0.02, 0.02, 0.25, 0.24):
            gui.text(f"Solver: {solver_names[solver]}")
            if gui.button("Explicit Euler"):
                solver = 0
                reset_simulation()
            if gui.button("Semi-Implicit Euler"):
                solver = 1
                reset_simulation()
            if gui.button("Implicit Euler"):
                solver = 2
                reset_simulation()
            damping = gui.slider_float("Damping", damping, 0.0, 8.0)

        for _ in range(PHYSICS_SUBSTEPS):
            if solver == 0:
                step_explicit(
                    DT, SPRING_STIFFNESS, damping, MAX_VELOCITY,
                    USE_SHEAR_SPRINGS, USE_BENDING_SPRINGS, ENABLE_SPHERE_COLLISION
                )
            elif solver == 1:
                step_semi_implicit(
                    DT, SPRING_STIFFNESS, damping, MAX_VELOCITY,
                    USE_SHEAR_SPRINGS, USE_BENDING_SPRINGS, ENABLE_SPHERE_COLLISION
                )
            else:
                step_implicit_iter(
                    DT, SPRING_STIFFNESS, damping, MAX_VELOCITY,
                    USE_SHEAR_SPRINGS, USE_BENDING_SPRINGS, ENABLE_SPHERE_COLLISION,
                    IMPLICIT_ITERATIONS
                )

        scene.set_camera(camera)
        scene.ambient_light((0.35, 0.35, 0.35))
        scene.point_light(pos=(2.5, 3.0, 3.0), color=(1.0, 1.0, 1.0))
        scene.mesh(
            positions,
            indices=triangle_indices,
            color=(0.18, 0.47, 0.86),
            two_sided=True,
        )
        scene.lines(positions, indices=line_indices, color=(0.08, 0.12, 0.18), width=1.0)
        scene.particles(positions, radius=0.012, color=(1.0, 0.86, 0.24))
        scene.particles(ball_position, radius=BALL_RADIUS, color=(0.9, 0.36, 0.22))
        canvas.scene(scene)
        window.show()


if __name__ == "__main__":
    main()
