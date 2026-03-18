"""Render CSP generative trajectories from .pt files as PNG sequences."""

import argparse
from pathlib import Path

import torch
from ase import Atoms
from ase.cell import Cell
from ase.io import write
from ase.io.utils import PlottingVariables


def load_trajectory(pt_path: Path) -> list[Atoms]:
    """Load a .pt trajectory file and return a list of ASE Atoms objects."""
    data = torch.load(pt_path, map_location="cpu", weights_only=True)
    frac_pred = data["frac_pred"].numpy()
    lengths = data["lengths"].numpy()
    angles = data["angles"].numpy()
    atom_types = data["atom_types"].numpy()

    frames = []
    for i in range(len(frac_pred)):
        cellpar = [*lengths[i], *angles[i]]
        cell = Cell.fromcellpar(cellpar)
        atoms = Atoms(
            numbers=atom_types[i],
            scaled_positions=frac_pred[i],
            cell=cell,
            pbc=True,
        )
        frames.append(atoms)
    return frames


def prepare_frame(atoms: Atoms, repeat: tuple[int, int, int]) -> Atoms:
    """Repeat atoms and restore original cell so unit cell lines match one period."""
    if repeat == (1, 1, 1):
        result = atoms.copy()
    else:
        result = atoms.repeat(repeat)
        result.set_cell(atoms.get_cell())
    # Center the original cell at the origin so the structure
    # stays centered as cell dimensions change across frames
    cell_center = atoms.get_cell().sum(axis=0) / 2
    result.translate(-cell_center)
    return result


def compute_canvas_size(all_trajectories: dict[int, list[Atoms]], repeat: tuple[int, int, int],
                        rotation: str, scale: int, show_unit_cell: int) -> tuple[float, float]:
    """Compute the canvas size (width, height) needed for all trajectories.

    Only checks the first and last frame of each trajectory since cell dimensions
    vary monotonically during generation. Takes the max per-frame width and height
    rather than the union of positions, producing a tighter canvas.
    """
    max_w = 0.0
    max_h = 0.0
    for frames in all_trajectories.values():
        for atoms in (frames[0], frames[-1]):
            prepared = prepare_frame(atoms, repeat)
            pvars = PlottingVariables(prepared, rotation=rotation, scale=scale, show_unit_cell=show_unit_cell)
            bbox = pvars.get_bbox()  # [xlo, ylo, xhi, yhi]
            max_w = max(max_w, bbox[2] - bbox[0])
            max_h = max(max_h, bbox[3] - bbox[1])
    # Add padding (2% on each side)
    max_w *= 1.04
    max_h *= 1.04
    return max_w, max_h


def render_frames(
    frames: list[Atoms],
    output_dir: Path,
    repeat: tuple[int, int, int] = (1, 1, 1),
    rotation: str = "",
    scale: int = 20,
    show_unit_cell: int = 2,
    canvas_w: float = 0.0,
    canvas_h: float = 0.0,
) -> None:
    """Render a list of Atoms frames as PNGs.

    When canvas_w and canvas_h are provided, each frame is centered within a
    fixed-size canvas by computing per-frame projected center coordinates.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    for i, atoms in enumerate(frames):
        prepared = prepare_frame(atoms, repeat)
        kwargs = {}
        if canvas_w > 0 and canvas_h > 0:
            pvars = PlottingVariables(prepared, rotation=rotation, scale=scale, show_unit_cell=show_unit_cell)
            frame_bbox = pvars.get_bbox()
            cx = (frame_bbox[0] + frame_bbox[2]) / 2
            cy = (frame_bbox[1] + frame_bbox[3]) / 2
            kwargs["bbox"] = [cx - canvas_w / 2, cy - canvas_h / 2, cx + canvas_w / 2, cy + canvas_h / 2]
        out_path = output_dir / f"frame_{i:03d}.png"
        write(
            str(out_path),
            prepared,
            show_unit_cell=show_unit_cell,
            rotation=rotation,
            scale=scale,
            **kwargs,
        )
    print(f"  Wrote {len(frames)} frames to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Render CSP trajectory PNGs")
    parser.add_argument("--traj_ids", nargs="+", type=int, default=[0], help="Trajectory IDs to render")
    parser.add_argument("--repeat", nargs=3, type=int, default=[2, 2, 2], help="Cell repetition (nx ny nz)")
    parser.add_argument("--rotation", type=str, default="", help="ASE rotation string")
    parser.add_argument("--scale", type=int, default=20, help="Scale factor (px/Å)")
    parser.add_argument("--show_unit_cell", type=int, default=2, choices=[0, 1, 2], help="Unit cell display mode")
    parser.add_argument("--output_dir", type=str, default="sgfm_traj/frames", help="Base output directory")
    parser.add_argument("--data_dir", type=str, default="sgfm_traj/trajectories/csp", help="Input data directory")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_base = Path(args.output_dir)
    repeat = tuple(args.repeat)

    # Load all trajectories
    trajectories = {}
    for traj_id in args.traj_ids:
        pt_path = data_dir / f"{traj_id}.pt"
        if not pt_path.exists():
            print(f"Warning: {pt_path} not found, skipping")
            continue
        trajectories[traj_id] = load_trajectory(pt_path)

    # Compute canvas size across all frames so every image has the same dimensions
    print("Computing canvas size...")
    canvas_w, canvas_h = compute_canvas_size(trajectories, repeat, args.rotation, args.scale, args.show_unit_cell)
    img_w = canvas_w * args.scale
    img_h = canvas_h * args.scale
    print(f"  canvas={canvas_w:.1f}x{canvas_h:.1f}, image size ~{img_w:.0f}x{img_h:.0f}px")

    for traj_id, frames in trajectories.items():
        print(f"Rendering trajectory {traj_id}")
        render_frames(
            frames,
            output_dir=output_base / f"traj_{traj_id}",
            repeat=repeat,
            rotation=args.rotation,
            scale=args.scale,
            show_unit_cell=args.show_unit_cell,
            canvas_w=canvas_w,
            canvas_h=canvas_h,
        )


if __name__ == "__main__":
    main()
