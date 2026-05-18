from __future__ import annotations

import argparse
import csv
import math
import os
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from pytorch3d.io import load_objs_as_meshes, save_obj
from pytorch3d.loss import (
    mesh_edge_loss,
    mesh_laplacian_smoothing,
    mesh_normal_consistency,
)
from pytorch3d.renderer import (
    BlendParams,
    FoVPerspectiveCameras,
    Materials,
    MeshRasterizer,
    MeshRenderer,
    PointLights,
    RasterizationSettings,
    SoftPhongShader,
    SoftSilhouetteShader,
    TexturesVertex,
    look_at_view_transform,
)
from pytorch3d.structures import Meshes
from pytorch3d.utils import ico_sphere


COW_ASSET_URLS = {
    "cow.obj": "https://dl.fbaipublicfiles.com/pytorch3d/data/cow_mesh/cow.obj",
    "cow.mtl": "https://dl.fbaipublicfiles.com/pytorch3d/data/cow_mesh/cow.mtl",
    "cow_texture.png": "https://dl.fbaipublicfiles.com/pytorch3d/data/cow_mesh/cow_texture.png",
}


@dataclass
class LossWeights:
    lap: float
    edge: float
    normal: float
    rgb: float
    color_smooth: float


def select_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def download_cow_assets(data_dir: Path) -> Path:
    data_dir.mkdir(parents=True, exist_ok=True)
    for filename, url in COW_ASSET_URLS.items():
        target = data_dir / filename
        if target.exists():
            continue
        print(f"Downloading {filename} ...")
        try:
            urllib.request.urlretrieve(url, target)
        except Exception as exc:
            raise RuntimeError(
                "Could not download the cow mesh automatically. "
                "Please download cow.obj/cow.mtl/cow_texture.png from the "
                "PyTorch3D tutorial data folder, or pass --target_obj manually."
            ) from exc
    return data_dir / "cow.obj"


def resolve_target_obj(target_obj: Path | None, data_dir: Path, no_download: bool) -> Path:
    if target_obj is not None:
        if not target_obj.exists():
            raise FileNotFoundError(f"Target obj not found: {target_obj}")
        return target_obj

    default_obj = data_dir / "cow.obj"
    if default_obj.exists():
        return default_obj
    if no_download:
        raise FileNotFoundError(
            f"{default_obj} does not exist. Pass --target_obj or remove --no_download."
        )
    return download_cow_assets(data_dir)


def build_cameras(
    device: torch.device,
    n_views: int,
    dist: float,
    elev: float,
) -> FoVPerspectiveCameras:
    azim = torch.linspace(0, 360, n_views + 1, device=device)[:-1]
    elev_t = torch.full_like(azim, elev)
    dist_t = torch.full_like(azim, dist)
    r, t = look_at_view_transform(dist=dist_t, elev=elev_t, azim=azim)
    return FoVPerspectiveCameras(device=device, R=r, T=t)


def sigmoid_blur_radius(sigma: float, cutoff: float = 1e-4) -> float:
    return float(math.log(1.0 / cutoff - 1.0) * sigma)


def build_silhouette_renderer(
    image_size: int,
    sigma: float,
    faces_per_pixel: int,
    cameras: FoVPerspectiveCameras,
) -> MeshRenderer:
    raster_settings = RasterizationSettings(
        image_size=image_size,
        blur_radius=sigmoid_blur_radius(sigma),
        faces_per_pixel=faces_per_pixel,
    )
    blend_params = BlendParams(sigma=sigma, gamma=1e-4)
    return MeshRenderer(
        rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings),
        shader=SoftSilhouetteShader(blend_params=blend_params),
    )


def build_rgb_renderer(
    image_size: int,
    sigma: float,
    faces_per_pixel: int,
    cameras: FoVPerspectiveCameras,
    device: torch.device,
) -> MeshRenderer:
    raster_settings = RasterizationSettings(
        image_size=image_size,
        blur_radius=sigmoid_blur_radius(sigma),
        faces_per_pixel=faces_per_pixel,
    )
    blend_params = BlendParams(sigma=sigma, gamma=1e-4, background_color=(0, 0, 0))
    lights = PointLights(device=device, location=[[0.0, 1.8, -3.0]])
    materials = Materials(
        device=device,
        ambient_color=((0.55, 0.55, 0.55),),
        diffuse_color=((0.55, 0.55, 0.55),),
        specular_color=((0.05, 0.05, 0.05),),
        shininess=16.0,
    )
    return MeshRenderer(
        rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings),
        shader=SoftPhongShader(
            device=device,
            cameras=cameras,
            lights=lights,
            materials=materials,
            blend_params=blend_params,
        ),
    )


def normalize_mesh(mesh: Meshes, target_scale: float = 1.7) -> Meshes:
    verts = mesh.verts_packed()
    min_xyz = verts.min(dim=0).values
    max_xyz = verts.max(dim=0).values
    center = (min_xyz + max_xyz) * 0.5
    scale = (max_xyz - min_xyz).max().clamp_min(1e-6)
    mesh = mesh.offset_verts(-center.expand_as(verts))
    return mesh.scale_verts(float((target_scale / scale).item()))


def procedural_cow_vertex_colors(mesh: Meshes) -> torch.Tensor:
    verts = mesh.verts_packed()
    x, y, z = verts.unbind(dim=1)
    spot_signal = (
        torch.sin(6.0 * x + 2.5 * z)
        + torch.sin(8.0 * y - 1.5 * x)
        + 0.7 * torch.sin(10.0 * z + 0.8)
    )
    spots = torch.sigmoid(6.0 * (spot_signal - 0.55)).unsqueeze(1)
    base = torch.tensor([0.90, 0.83, 0.68], device=verts.device)
    dark = torch.tensor([0.06, 0.055, 0.05], device=verts.device)
    colors = base * (1.0 - spots) + dark * spots
    underside = torch.sigmoid(-10.0 * (y + 0.55)).unsqueeze(1)
    hoof = torch.tensor([0.03, 0.025, 0.02], device=verts.device)
    colors = colors * (1.0 - 0.45 * underside) + hoof * (0.45 * underside)
    return colors.clamp(0.0, 1.0).unsqueeze(0)


def mesh_has_texture(mesh: Meshes) -> bool:
    return mesh.textures is not None


def with_vertex_colors(mesh: Meshes, colors: torch.Tensor) -> Meshes:
    return Meshes(
        verts=mesh.verts_list(),
        faces=mesh.faces_list(),
        textures=TexturesVertex(verts_features=colors),
    )


def load_target_mesh(
    target_obj: Path,
    device: torch.device,
    target_texture: str,
) -> Meshes:
    mesh = load_objs_as_meshes([str(target_obj)], device=device)
    mesh = normalize_mesh(mesh)
    if target_texture == "procedural" or not mesh_has_texture(mesh):
        mesh = with_vertex_colors(mesh, procedural_cow_vertex_colors(mesh))
    return mesh


def save_image_grid(
    images: torch.Tensor,
    out_path: Path,
    n_cols: int = 4,
    title: str | None = None,
    cmap: str | None = None,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    images = images.detach().cpu().clamp(0.0, 1.0)
    n = images.shape[0]
    n_rows = (n + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.0 * n_cols, 3.0 * n_rows))
    axes = axes.reshape(-1) if hasattr(axes, "reshape") else [axes]
    for i in range(n_rows * n_cols):
        ax = axes[i]
        ax.axis("off")
        if i >= n:
            continue
        image = images[i].numpy()
        if image.ndim == 2:
            ax.imshow(image, cmap=cmap or "gray", vmin=0.0, vmax=1.0)
        else:
            ax.imshow(image)
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def save_loss_csv(loss_rows: list[dict[str, float]], out_path: Path) -> None:
    if not loss_rows:
        return
    keys = list(loss_rows[0].keys())
    with out_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=keys)
        writer.writeheader()
        writer.writerows(loss_rows)


def save_loss_curve(loss_rows: list[dict[str, float]], out_path: Path) -> None:
    if not loss_rows:
        return
    steps = [row["iter"] for row in loss_rows]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for key in ["total", "silhouette", "rgb", "laplacian", "edge", "normal"]:
        if key in loss_rows[0]:
            ax.plot(steps, [row[key] for row in loss_rows], label=key)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Loss")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def make_progress_gif(output_dir: Path, mode: str) -> None:
    try:
        import imageio.v2 as imageio
    except Exception:
        print("imageio is not installed; skip GIF export.")
        return

    pattern = "pred_rgb_iter_*.png" if mode == "joint" else "pred_sil_iter_*.png"
    frame_paths = sorted(output_dir.glob(pattern))
    if len(frame_paths) < 2:
        return
    frames = [imageio.imread(path) for path in frame_paths]
    gif_path = output_dir / ("joint_rgb_progress.gif" if mode == "joint" else "silhouette_progress.gif")
    imageio.mimsave(gif_path, frames, duration=0.45)
    print(f"Saved progress GIF: {gif_path}")


def vertex_color_smoothness(mesh: Meshes, colors: torch.Tensor) -> torch.Tensor:
    edges = mesh.edges_packed()
    if edges.numel() == 0:
        return colors.new_tensor(0.0)
    rgb = colors[0]
    return (rgb[edges[:, 0]] - rgb[edges[:, 1]]).pow(2).mean()


def build_source_mesh(
    source_mesh: Meshes,
    deform_verts: torch.Tensor,
    color_logits: torch.Tensor | None,
) -> Meshes:
    deformed = source_mesh.offset_verts(deform_verts)
    if color_logits is None:
        return deformed
    vertex_colors = torch.sigmoid(color_logits)
    return with_vertex_colors(deformed, vertex_colors)


def save_colored_obj(path: Path, mesh: Meshes, vertex_colors: torch.Tensor) -> None:
    verts = mesh.verts_packed().detach().cpu()
    faces = mesh.faces_packed().detach().cpu()
    colors = vertex_colors[0].detach().cpu().clamp(0.0, 1.0)
    with path.open("w", encoding="utf-8") as file:
        file.write("# OBJ with per-vertex RGB colors\n")
        for vert, color in zip(verts, colors):
            file.write(
                "v "
                f"{vert[0].item():.7f} {vert[1].item():.7f} {vert[2].item():.7f} "
                f"{color[0].item():.7f} {color[1].item():.7f} {color[2].item():.7f}\n"
            )
        for face in faces:
            file.write(
                f"f {face[0].item() + 1} {face[1].item() + 1} {face[2].item() + 1}\n"
            )


def render_reference_images(
    target_mesh: Meshes,
    n_views: int,
    cameras: FoVPerspectiveCameras,
    silhouette_renderer: MeshRenderer,
    rgb_renderer: MeshRenderer | None,
    output_dir: Path,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    with torch.no_grad():
        target_sil = silhouette_renderer(target_mesh.extend(n_views), cameras=cameras)[..., 3]
        save_image_grid(
            target_sil,
            output_dir / "target_silhouettes.png",
            title="Target silhouettes",
        )
        target_rgb = None
        if rgb_renderer is not None:
            target_rgb = rgb_renderer(target_mesh.extend(n_views), cameras=cameras)[..., :3]
            save_image_grid(
                target_rgb,
                output_dir / "target_rgb.png",
                title="Target RGB views",
            )
    return target_sil.detach(), None if target_rgb is None else target_rgb.detach()


def snapshot_outputs(
    iteration: int,
    mesh: Meshes,
    color_logits: torch.Tensor | None,
    n_views: int,
    cameras: FoVPerspectiveCameras,
    silhouette_renderer: MeshRenderer,
    rgb_renderer: MeshRenderer | None,
    output_dir: Path,
) -> None:
    with torch.no_grad():
        pred_sil = silhouette_renderer(mesh.extend(n_views), cameras=cameras)[..., 3]
        save_image_grid(
            pred_sil,
            output_dir / f"pred_sil_iter_{iteration:04d}.png",
            title=f"Silhouette prediction @ iter {iteration}",
        )
        if rgb_renderer is not None:
            pred_rgb = rgb_renderer(mesh.extend(n_views), cameras=cameras)[..., :3]
            save_image_grid(
                pred_rgb,
                output_dir / f"pred_rgb_iter_{iteration:04d}.png",
                title=f"RGB prediction @ iter {iteration}",
            )

        save_obj(
            f=str(output_dir / f"mesh_iter_{iteration:04d}.obj"),
            verts=mesh.verts_packed().detach().cpu(),
            faces=mesh.faces_packed().detach().cpu(),
        )
        if color_logits is not None:
            save_colored_obj(
                output_dir / f"colored_mesh_iter_{iteration:04d}.obj",
                mesh,
                torch.sigmoid(color_logits.detach()),
            )


def optimize_mesh(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

    device = select_device(args.device)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    target_obj = resolve_target_obj(args.target_obj, args.data_dir, args.no_download)
    print(f"Device: {device}")
    print(f"Target mesh: {target_obj}")
    print(f"Mode: {args.mode}")

    cameras = build_cameras(
        device=device,
        n_views=args.n_views,
        dist=args.camera_dist,
        elev=args.camera_elev,
    )
    silhouette_renderer = build_silhouette_renderer(
        image_size=args.image_size,
        sigma=args.sigma,
        faces_per_pixel=args.faces_per_pixel,
        cameras=cameras,
    )
    rgb_renderer = None
    if args.mode == "joint":
        rgb_renderer = build_rgb_renderer(
            image_size=args.image_size,
            sigma=args.rgb_sigma,
            faces_per_pixel=args.faces_per_pixel,
            cameras=cameras,
            device=device,
        )

    target_mesh = load_target_mesh(target_obj, device, args.target_texture)
    target_alpha, target_rgb = render_reference_images(
        target_mesh=target_mesh,
        n_views=args.n_views,
        cameras=cameras,
        silhouette_renderer=silhouette_renderer,
        rgb_renderer=rgb_renderer,
        output_dir=output_dir,
    )

    source_mesh = ico_sphere(level=args.ico_level, device=device)
    deform_verts = torch.zeros_like(source_mesh.verts_packed(), requires_grad=True)
    color_logits = None
    params: list[torch.Tensor] = [deform_verts]
    if args.mode == "joint":
        n_verts = source_mesh.verts_packed().shape[0]
        init_rgb = torch.full((1, n_verts, 3), 0.65, device=device)
        color_logits = torch.logit(init_rgb, eps=1e-4).detach().requires_grad_(True)
        params.append(color_logits)

    weights = LossWeights(
        lap=args.w_lap,
        edge=args.w_edge,
        normal=args.w_normal,
        rgb=args.w_rgb,
        color_smooth=args.w_color_smooth,
    )
    optimizer = torch.optim.Adam(params, lr=args.lr)
    all_view_indices = torch.arange(args.n_views, device=device)
    loss_rows: list[dict[str, float]] = []

    initial_mesh = build_source_mesh(
        source_mesh,
        deform_verts.detach(),
        None if color_logits is None else color_logits.detach(),
    )
    snapshot_outputs(
        iteration=0,
        mesh=initial_mesh,
        color_logits=None if color_logits is None else color_logits.detach(),
        n_views=args.n_views,
        cameras=cameras,
        silhouette_renderer=silhouette_renderer,
        rgb_renderer=rgb_renderer,
        output_dir=output_dir,
    )

    for iteration in range(1, args.n_iter + 1):
        optimizer.zero_grad()
        if 0 < args.views_per_iter < args.n_views:
            view_indices = all_view_indices[torch.randperm(args.n_views, device=device)[: args.views_per_iter]]
        else:
            view_indices = all_view_indices

        batch_cameras = cameras[view_indices]
        current_mesh = build_source_mesh(source_mesh, deform_verts, color_logits)
        batch_mesh = current_mesh.extend(len(view_indices))

        pred_alpha = silhouette_renderer(batch_mesh, cameras=batch_cameras)[..., 3]
        loss_sil = F.mse_loss(pred_alpha, target_alpha[view_indices])
        loss_lap = mesh_laplacian_smoothing(current_mesh, method="uniform")
        loss_edge = mesh_edge_loss(current_mesh)
        loss_normal = mesh_normal_consistency(current_mesh)
        loss_rgb = deform_verts.new_tensor(0.0)
        loss_color = deform_verts.new_tensor(0.0)

        if args.mode == "joint" and rgb_renderer is not None and target_rgb is not None:
            pred_rgb = rgb_renderer(batch_mesh, cameras=batch_cameras)[..., :3]
            rgb_mask = target_alpha[view_indices].unsqueeze(-1).clamp(0.0, 1.0)
            loss_rgb = ((pred_rgb - target_rgb[view_indices]) * rgb_mask).pow(2).mean()
            loss_color = vertex_color_smoothness(current_mesh, torch.sigmoid(color_logits))

        active_rgb_weight = weights.rgb if iteration >= args.rgb_start_iter else 0.0
        loss = (
            loss_sil
            + weights.lap * loss_lap
            + weights.edge * loss_edge
            + weights.normal * loss_normal
            + active_rgb_weight * loss_rgb
            + weights.color_smooth * loss_color
        )
        loss.backward()
        optimizer.step()

        if iteration % args.log_every == 0 or iteration == 1 or iteration == args.n_iter:
            row = {
                "iter": float(iteration),
                "total": float(loss.item()),
                "silhouette": float(loss_sil.item()),
                "rgb": float(loss_rgb.item()),
                "laplacian": float(loss_lap.item()),
                "edge": float(loss_edge.item()),
                "normal": float(loss_normal.item()),
                "color_smooth": float(loss_color.item()),
            }
            loss_rows.append(row)
            print(
                f"[{iteration:04d}/{args.n_iter}] "
                f"total={row['total']:.6f} sil={row['silhouette']:.6f} "
                f"rgb={row['rgb']:.6f} lap={row['laplacian']:.6f} "
                f"edge={row['edge']:.6f} normal={row['normal']:.6f}"
            )

        if iteration % args.save_every == 0 or iteration == args.n_iter:
            snapshot_mesh = build_source_mesh(
                source_mesh,
                deform_verts.detach(),
                None if color_logits is None else color_logits.detach(),
            )
            snapshot_outputs(
                iteration=iteration,
                mesh=snapshot_mesh,
                color_logits=None if color_logits is None else color_logits.detach(),
                n_views=args.n_views,
                cameras=cameras,
                silhouette_renderer=silhouette_renderer,
                rgb_renderer=rgb_renderer,
                output_dir=output_dir,
            )

    final_mesh = build_source_mesh(
        source_mesh,
        deform_verts.detach(),
        None if color_logits is None else color_logits.detach(),
    )
    save_obj(
        f=str(output_dir / "optimized_mesh.obj"),
        verts=final_mesh.verts_packed().detach().cpu(),
        faces=final_mesh.faces_packed().detach().cpu(),
    )
    if color_logits is not None:
        final_colors = torch.sigmoid(color_logits.detach())
        save_colored_obj(output_dir / "optimized_colored_mesh.obj", final_mesh, final_colors)
        torch.save(
            {
                "verts": final_mesh.verts_packed().detach().cpu(),
                "faces": final_mesh.faces_packed().detach().cpu(),
                "vertex_colors": final_colors.detach().cpu(),
            },
            output_dir / "optimized_colored_mesh.pt",
        )

    save_loss_csv(loss_rows, output_dir / "losses.csv")
    save_loss_curve(loss_rows, output_dir / "loss_curve.png")
    if args.make_gif:
        make_progress_gif(output_dir, args.mode)
    print(f"\nDone. Results saved to: {output_dir.resolve()}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CG Lab 6: differentiable rendering with silhouette and RGB fitting."
    )
    parser.add_argument("--target_obj", type=Path, default=None, help="Path to target cow .obj.")
    parser.add_argument("--data_dir", type=Path, default=Path("data/cow_mesh"))
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--mode", choices=["silhouette", "joint"], default="joint")
    parser.add_argument("--target_texture", choices=["auto", "procedural"], default="auto")
    parser.add_argument("--no_download", action="store_true")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda:0, ...")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--image_size", type=int, default=128)
    parser.add_argument("--n_views", type=int, default=16)
    parser.add_argument("--views_per_iter", type=int, default=8)
    parser.add_argument("--n_iter", type=int, default=800)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--ico_level", type=int, default=4)
    parser.add_argument("--camera_dist", type=float, default=2.7)
    parser.add_argument("--camera_elev", type=float, default=20.0)

    parser.add_argument("--sigma", type=float, default=1e-4)
    parser.add_argument("--rgb_sigma", type=float, default=1e-4)
    parser.add_argument("--faces_per_pixel", type=int, default=50)
    parser.add_argument("--w_lap", type=float, default=0.08)
    parser.add_argument("--w_edge", type=float, default=0.8)
    parser.add_argument("--w_normal", type=float, default=0.01)
    parser.add_argument("--w_rgb", type=float, default=0.5)
    parser.add_argument("--w_color_smooth", type=float, default=0.03)
    parser.add_argument("--rgb_start_iter", type=int, default=1)

    parser.add_argument("--log_every", type=int, default=20)
    parser.add_argument("--save_every", type=int, default=100)
    parser.add_argument("--make_gif", action="store_true")
    args = parser.parse_args()
    if args.output_dir is None:
        args.output_dir = Path("outputs") / args.mode
    return args


if __name__ == "__main__":
    optimize_mesh(parse_args())
