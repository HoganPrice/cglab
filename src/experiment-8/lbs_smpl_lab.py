from __future__ import annotations

import argparse
import sys
import types
from pathlib import Path

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def require_imageio():
    try:
        import imageio.v2 as imageio
    except ImportError as exc:
        raise SystemExit("缺少依赖 imageio，请先运行 pip install -r requirements.txt") from exc
    return imageio


def ensure_chumpy_compat() -> None:
    try:
        import chumpy  # noqa: F401
        return
    except ImportError:
        pass

    chumpy_module = types.ModuleType("chumpy")
    ch_module = types.ModuleType("chumpy.ch")

    class Ch:
        def __setstate__(self, state):
            self.__dict__.update(state)

        def __getattr__(self, name):
            return getattr(self.x, name)

        def __getitem__(self, key):
            return self.x[key]

        @property
        def r(self):
            return self.x

        def __array__(self, dtype=None):
            return np.asarray(self.x, dtype=dtype)

    Ch.__module__ = "chumpy.ch"
    ch_module.Ch = Ch
    chumpy_module.ch = ch_module
    sys.modules["chumpy"] = chumpy_module
    sys.modules["chumpy.ch"] = ch_module


def batch_rodrigues(rot_vecs: torch.Tensor) -> torch.Tensor:
    """Convert axis-angle vectors to rotation matrices."""
    dtype = rot_vecs.dtype
    device = rot_vecs.device
    batch_size = rot_vecs.shape[0]
    angle = torch.norm(rot_vecs + 1e-8, dim=1, keepdim=True)
    direction = rot_vecs / angle
    cos = torch.cos(angle).view(-1, 1, 1)
    sin = torch.sin(angle).view(-1, 1, 1)

    rx, ry, rz = torch.split(direction, 1, dim=1)
    zeros = torch.zeros((batch_size, 1), dtype=dtype, device=device)
    k = torch.cat(
        [
            zeros,
            -rz,
            ry,
            rz,
            zeros,
            -rx,
            -ry,
            rx,
            zeros,
        ],
        dim=1,
    ).view(batch_size, 3, 3)
    ident = torch.eye(3, dtype=dtype, device=device).unsqueeze(0)
    return ident + sin * k + (1.0 - cos) * torch.bmm(k, k)


def make_transform(rot: torch.Tensor, trans: torch.Tensor) -> torch.Tensor:
    batch_size = rot.shape[0]
    transform = torch.zeros((batch_size, 4, 4), dtype=rot.dtype, device=rot.device)
    transform[:, :3, :3] = rot
    transform[:, :3, 3] = trans
    transform[:, 3, 3] = 1.0
    return transform


def batch_rigid_transform(
    rot_mats: torch.Tensor,
    joints: torch.Tensor,
    parents: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size, num_joints = joints.shape[:2]
    joints_homogen = joints.clone()
    joints_homogen[:, 1:] -= joints[:, parents[1:]]

    rel_transforms = make_transform(
        rot_mats.reshape(-1, 3, 3),
        joints_homogen.reshape(-1, 3),
    ).view(batch_size, num_joints, 4, 4)

    transform_chain = [rel_transforms[:, 0]]
    for joint_id in range(1, num_joints):
        parent_id = int(parents[joint_id])
        transform_chain.append(torch.matmul(transform_chain[parent_id], rel_transforms[:, joint_id]))
    transforms = torch.stack(transform_chain, dim=1)

    joints_h = torch.cat(
        [joints, torch.zeros(batch_size, num_joints, 1, dtype=joints.dtype, device=joints.device)],
        dim=2,
    ).unsqueeze(-1)
    init_bone = torch.matmul(transforms, joints_h)
    init_bone = torch.nn.functional.pad(init_bone, [3, 0, 0, 0, 0, 0, 0, 0])
    rel_transforms = transforms - init_bone
    posed_joints = transforms[:, :, :3, 3]
    return posed_joints, rel_transforms


def blend_shapes(betas: torch.Tensor, shape_disps: torch.Tensor) -> torch.Tensor:
    return torch.einsum("bl,mkl->bmk", [betas, shape_disps])


def vertices2joints(j_regressor: torch.Tensor, vertices: torch.Tensor) -> torch.Tensor:
    return torch.einsum("ji,bik->bjk", [j_regressor, vertices])


def manual_lbs(
    betas: torch.Tensor,
    pose: torch.Tensor,
    model,
) -> dict[str, torch.Tensor]:
    v_template = model.v_template.unsqueeze(0).to(device=betas.device, dtype=betas.dtype)
    shapedirs = model.shapedirs.to(device=betas.device, dtype=betas.dtype)
    posedirs = model.posedirs.to(device=betas.device, dtype=betas.dtype)
    j_regressor = model.J_regressor.to(device=betas.device, dtype=betas.dtype)
    parents = model.parents.to(device=betas.device)
    lbs_weights = model.lbs_weights.to(device=betas.device, dtype=betas.dtype)

    batch_size = betas.shape[0]
    v_shaped = v_template + blend_shapes(betas, shapedirs)
    joints = vertices2joints(j_regressor, v_shaped)

    rot_mats = batch_rodrigues(pose.reshape(-1, 3)).view(batch_size, -1, 3, 3)
    ident = torch.eye(3, dtype=betas.dtype, device=betas.device)
    pose_feature = (rot_mats[:, 1:] - ident).reshape(batch_size, -1)
    pose_offsets = torch.matmul(pose_feature, posedirs).view(batch_size, -1, 3)
    v_posed = v_shaped + pose_offsets

    j_transformed, transforms = batch_rigid_transform(rot_mats, joints, parents)
    weights = lbs_weights.unsqueeze(0).expand(batch_size, -1, -1)
    num_joints = joints.shape[1]
    vertex_transforms = torch.matmul(weights, transforms.reshape(batch_size, num_joints, 16))
    vertex_transforms = vertex_transforms.view(batch_size, -1, 4, 4)

    v_posed_homo = torch.cat(
        [v_posed, torch.ones(batch_size, v_posed.shape[1], 1, dtype=betas.dtype, device=betas.device)],
        dim=2,
    )
    v_homo = torch.matmul(vertex_transforms, v_posed_homo.unsqueeze(-1))
    verts = v_homo[:, :, :3, 0]

    return {
        "v_template": v_template,
        "v_shaped": v_shaped,
        "joints": joints,
        "rot_mats": rot_mats,
        "pose_feature": pose_feature,
        "pose_offsets": pose_offsets,
        "v_posed": v_posed,
        "j_transformed": j_transformed,
        "transforms": transforms,
        "verts": verts,
    }


def as_numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy()


def to_plot_coords(points: np.ndarray) -> np.ndarray:
    # SMPL uses Y as the vertical axis, while Matplotlib 3D displays Z as up.
    return points[:, [0, 2, 1]]


def set_axes_equal(ax, vertices: np.ndarray) -> None:
    mins = vertices.min(axis=0)
    maxs = vertices.max(axis=0)
    center = (mins + maxs) * 0.5
    radius = (maxs - mins).max() * 0.55
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)


def render_mesh(
    vertices: np.ndarray,
    faces: np.ndarray,
    out_path: Path,
    title: str,
    vertex_values: np.ndarray | None = None,
    joints: np.ndarray | None = None,
    cmap_name: str = "viridis",
    elev: float = 10.0,
    azim: float = 110.0,
) -> None:
    fig = plt.figure(figsize=(7, 7), dpi=160)
    ax = fig.add_subplot(111, projection="3d")
    plot_vertices = to_plot_coords(vertices)
    plot_joints = to_plot_coords(joints) if joints is not None else None
    triangles = plot_vertices[faces]

    if vertex_values is None:
        face_colors = np.tile(np.array([[0.68, 0.76, 0.84, 1.0]]), (faces.shape[0], 1))
    else:
        values = np.asarray(vertex_values)
        values = (values - values.min()) / (values.max() - values.min() + 1e-12)
        face_values = values[faces].mean(axis=1)
        face_colors = cm.get_cmap(cmap_name)(face_values)

    mesh = Poly3DCollection(triangles, facecolors=face_colors, linewidths=0.02, alpha=0.98)
    mesh.set_edgecolor((0.08, 0.08, 0.08, 0.08))
    ax.add_collection3d(mesh)

    if joints is not None:
        ax.scatter(plot_joints[:, 0], plot_joints[:, 1], plot_joints[:, 2], c="#e22d2d", s=22, depthshade=False)
        ax.plot(plot_joints[:, 0], plot_joints[:, 1], plot_joints[:, 2], color="#e22d2d", linewidth=0.8, alpha=0.65)

    set_axes_equal(ax, plot_vertices)
    ax.view_init(elev=elev, azim=azim)
    ax.set_title(title)
    ax.set_axis_off()
    fig.tight_layout(pad=0.1)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def make_comparison_grid(image_paths: list[Path], out_path: Path) -> None:
    imageio = require_imageio()
    images = [imageio.imread(path) for path in image_paths]
    min_h = min(img.shape[0] for img in images)
    min_w = min(img.shape[1] for img in images)
    images = [img[:min_h, :min_w, :3] for img in images]
    top = np.concatenate(images[:2], axis=1)
    bottom = np.concatenate(images[2:], axis=1)
    grid = np.concatenate([top, bottom], axis=0)
    imageio.imwrite(out_path, grid)


def make_pose(global_orient: torch.Tensor, body_pose: torch.Tensor) -> torch.Tensor:
    return torch.cat([global_orient.reshape(1, 1, 3), body_pose.reshape(1, 23, 3)], dim=1)


def build_demo_parameters(device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    betas = torch.zeros((1, 10), dtype=dtype, device=device)
    betas[0, 0] = 1.4
    betas[0, 1] = -0.8
    betas[0, 2] = 0.5

    global_orient = torch.zeros((1, 3), dtype=dtype, device=device)
    body_pose = torch.zeros((1, 23, 3), dtype=dtype, device=device)
    body_pose[0, 16, 2] = -0.85
    body_pose[0, 17, 2] = 0.85
    body_pose[0, 18, 0] = -0.55
    body_pose[0, 19, 0] = -0.55
    body_pose[0, 1, 0] = 0.35
    body_pose[0, 2, 0] = -0.25
    return betas, global_orient, body_pose


def create_model(model_dir: Path, device: torch.device, dtype: torch.dtype):
    ensure_chumpy_compat()
    try:
        import smplx
    except ImportError as exc:
        raise SystemExit("Missing dependency smplx. Please run: pip install -r requirements.txt") from exc

    model_dir = model_dir.resolve()
    candidates = [
        model_dir if model_dir.is_file() else None,
        model_dir / "SMPL_NEUTRAL.pkl",
        model_dir / "smpl" / "SMPL_NEUTRAL.pkl",
        model_dir / "SMPL" / "SMPL_NEUTRAL.pkl",
    ]
    model_path = next((path for path in candidates if path is not None and path.exists()), None)
    if model_path is None:
        expected = "\n".join(str(path) for path in candidates[1:])
        raise SystemExit(f"Could not find SMPL_NEUTRAL.pkl. Checked:\n{expected}")

    return smplx.SMPL(
        str(model_path),
        gender="neutral",
        num_betas=10,
        batch_size=1,
        dtype=dtype,
    ).to(device=device, dtype=dtype)


def write_summary(
    out_path: Path,
    model,
    results: dict[str, torch.Tensor],
    official_vertices: torch.Tensor,
) -> None:
    verts = results["verts"]
    diff = torch.abs(verts - official_vertices)
    text = "\n".join(
        [
            "Experiment 8 - SMPL LBS Summary",
            "",
            f"vertices: {model.v_template.shape[0]}",
            f"faces: {model.faces.shape[0]}",
            f"joints: {model.J_regressor.shape[0]}",
            f"betas dimension: {model.num_betas}",
            f"mean absolute error: {diff.mean().item():.10e}",
            f"max absolute error: {diff.max().item():.10e}",
            "",
            "outputs:",
            "- stage_a_template_weights.png",
            "- all_joint_weights.png",
            "- stage_b_shaped_joints.png",
            "- stage_c_pose_offsets.png",
            "- stage_d_lbs_result.png",
            "- comparison_grid.png",
            "- optional_pose_animation.gif",
        ]
    )
    out_path.write_text(text, encoding="utf-8")


def run_required_and_optional(args: argparse.Namespace) -> None:
    imageio = require_imageio()
    device = torch.device(args.device)
    dtype = torch.float32
    out_dir = Path(args.output_dir)
    model = create_model(Path(args.model_dir), device, dtype)
    faces = np.asarray(model.faces, dtype=np.int64)

    betas, global_orient, body_pose = build_demo_parameters(device, dtype)
    pose = make_pose(global_orient, body_pose)
    results = manual_lbs(betas, pose, model)

    with torch.no_grad():
        official = model(
            betas=betas,
            global_orient=global_orient,
            body_pose=body_pose.reshape(1, -1),
            return_verts=True,
        ).vertices

    lbs_weights = as_numpy(model.lbs_weights)
    selected_joint = args.joint
    dominant_joint = lbs_weights.argmax(axis=1).astype(np.float32)
    dominant_strength = lbs_weights.max(axis=1)
    all_joint_values = dominant_joint + dominant_strength

    paths = {
        "stage_a": out_dir / "stage_a_template_weights.png",
        "all_weights": out_dir / "all_joint_weights.png",
        "stage_b": out_dir / "stage_b_shaped_joints.png",
        "stage_c": out_dir / "stage_c_pose_offsets.png",
        "stage_d": out_dir / "stage_d_lbs_result.png",
        "grid": out_dir / "comparison_grid.png",
        "gif": out_dir / "optional_pose_animation.gif",
        "summary": out_dir / "summary.txt",
    }

    render_mesh(
        as_numpy(results["v_template"][0]),
        faces,
        paths["stage_a"],
        f"(a) template + joint {selected_joint} weights",
        vertex_values=lbs_weights[:, selected_joint],
    )
    render_mesh(
        as_numpy(results["v_template"][0]),
        faces,
        paths["all_weights"],
        "optional: dominant joint weights",
        vertex_values=all_joint_values,
        cmap_name="tab20",
    )
    render_mesh(
        as_numpy(results["v_shaped"][0]),
        faces,
        paths["stage_b"],
        "(b) shape + regressed joints",
        joints=as_numpy(results["joints"][0]),
    )
    pose_offset_norm = torch.norm(results["pose_offsets"][0], dim=1)
    render_mesh(
        as_numpy(results["v_posed"][0]),
        faces,
        paths["stage_c"],
        "(c) pose corrective offsets",
        vertex_values=as_numpy(pose_offset_norm),
        cmap_name="magma",
    )
    render_mesh(
        as_numpy(results["verts"][0]),
        faces,
        paths["stage_d"],
        "(d) final skinned mesh",
        joints=as_numpy(results["j_transformed"][0]),
    )
    make_comparison_grid(
        [paths["stage_a"], paths["stage_b"], paths["stage_c"], paths["stage_d"]],
        paths["grid"],
    )
    write_summary(paths["summary"], model, results, official)

    frames = []
    joint_axis = int(args.animation_joint)
    for frame_id, angle in enumerate(np.linspace(0.0, float(args.animation_angle), int(args.animation_frames))):
        anim_body_pose = body_pose.clone()
        anim_body_pose[0, joint_axis, 2] = torch.tensor(angle, dtype=dtype, device=device)
        anim_results = manual_lbs(betas, make_pose(global_orient, anim_body_pose), model)
        frame_path = out_dir / f"_anim_frame_{frame_id:03d}.png"
        render_mesh(
            as_numpy(anim_results["verts"][0]),
            faces,
            frame_path,
            f"optional pose animation frame {frame_id:02d}",
            joints=as_numpy(anim_results["j_transformed"][0]),
            elev=10,
            azim=105,
        )
        frames.append(imageio.imread(frame_path))
        frame_path.unlink(missing_ok=True)
    imageio.mimsave(paths["gif"], frames, duration=0.09)

    print(f"Done. Results written to {out_dir.resolve()}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Experiment 8: SMPL Linear Blend Skinning visualization")
    parser.add_argument("--model-dir", default="data", help="Directory containing SMPL_NEUTRAL.pkl")
    parser.add_argument("--output-dir", default="outputs", help="Directory for rendered images and summary.txt")
    parser.add_argument("--device", default="cpu", help="cpu or cuda")
    parser.add_argument("--joint", type=int, default=16, help="Joint index used for the single-joint weight heatmap")
    parser.add_argument("--animation-joint", type=int, default=16, help="Body-pose joint index for the optional animation")
    parser.add_argument("--animation-angle", type=float, default=1.2, help="Final Z-axis rotation in radians")
    parser.add_argument("--animation-frames", type=int, default=24, help="Number of frames in optional GIF")
    return parser.parse_args()


if __name__ == "__main__":
    run_required_and_optional(parse_args())
