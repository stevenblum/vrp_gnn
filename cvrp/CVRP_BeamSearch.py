"""
Run configurable CVRP inference (greedy, beam search, sampling) on CVRPLib instances.

Primary output: CSV with per-instance costs and gaps.
Helpers: terminal summary and heatmap PNG of optimality gaps.

python CVRP_BeamSearch.py \
  --checkpoint /home/scblum/Projects/vrp_gnn/cvrp/lightning_logs/small_exp1/emb512_enc5_attn8/version_4/checkpoints/emb512_enc5_attn8-epochepoch=012-valval/reward=-53.515.ckpt  \
  --beam_widths 10,100,1000,5000 \
  --beam_temps 1.5,2.0,3.0 \
  --beam_temp_width 1000 \
  --sampling_counts 1000,100000 \
  --sampling_temp_count 10000 \
  --sampling_temps 1.5,2.0,3.0 \
  --print_table

  
"""
from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
from typing import Dict, List, Sequence

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import torch
from rl4co.envs.routing import CVRPEnv, CVRPGenerator
from rl4co.models import AttentionModelPolicy, POMO
from tensordict import TensorDict

from classes.CVRPLibHelpers import calculate_normalized_bks, load_bks_cost, load_val_instance


# ----------------------------- CLI helpers ----------------------------- #


def _list_ints(arg: str) -> List[int]:
    if arg.strip() == "":
        return []
    return [int(x) for x in arg.split(",")]


def _list_floats(arg: str) -> List[float]:
    if arg.strip() == "":
        return []
    return [float(x) for x in arg.split(",")]


# ----------------------------- Core logic ----------------------------- #


def load_model(checkpoint_path: Path, device: torch.device) -> POMO:
    """Load POMO + AttentionModel from checkpoint (ignoring env keys)."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    hparams = checkpoint.get("hyper_parameters", {})

    embed_dim = hparams.get("embed_dim") or hparams.get("embedding_dim")
    num_encoder_layers = hparams.get("num_encoder_layers")
    num_heads = hparams.get("num_heads") or hparams.get("num_attn_heads")

    sd = checkpoint["state_dict"]
    if embed_dim is None:
        for k, v in sd.items():
            if "init_embedding.init_embed.weight" in k:
                embed_dim = v.shape[0]
                break
    if num_encoder_layers is None:
        layer_idxs = set()
        for k in sd.keys():
            if "encoder.net.layers." in k:
                try:
                    idx = int(k.split("encoder.net.layers.")[1].split(".")[0])
                    layer_idxs.add(idx)
                except Exception:
                    pass
        if layer_idxs:
            num_encoder_layers = max(layer_idxs) + 1
    if num_heads is None:
        num_heads = hparams.get("num_heads", 8)

    embed_dim = embed_dim or 256
    num_encoder_layers = num_encoder_layers or 6
    num_heads = num_heads or 8

    temp_env = CVRPEnv(CVRPGenerator(num_loc=100))
    policy = AttentionModelPolicy(
        env_name="cvrp",
        embed_dim=embed_dim,
        num_encoder_layers=num_encoder_layers,
        num_heads=num_heads,
        normalization="batch",
    )
    model = POMO(temp_env, policy)
    state_dict = {k: v for k, v in checkpoint["state_dict"].items() if not k.startswith("env.")}
    model.load_state_dict(state_dict, strict=False)
    model.to(device).eval()
    return model


def load_instances(instances: Sequence[str], base_dir: Path) -> List[TensorDict]:
    tds = []
    for stem in instances:
        npz = base_dir / f"{stem}.npz"
        vrp = base_dir.parent / "X" / f"{stem}.vrp"
        if npz.exists():
            tds.append(load_val_instance(str(npz)))
        elif vrp.exists():
            tds.append(load_val_instance(str(vrp)))
        else:
            raise FileNotFoundError(f"Couldn't find {stem} as .npz or .vrp in {base_dir} or {vrp.parent}")
    return tds


def discover_all_instances(base_dir: Path) -> List[str]:
    return sorted(p.stem for p in base_dir.glob("*.npz"))


def run_methods_on_instance(
    model: POMO,
    batch: TensorDict,
    beam_widths: Sequence[int],
    beam_temps: Sequence[float],
    beam_temp_width: int,
    sampling_counts: Sequence[int],
    sampling_temp_count: int,
    sampling_temps: Sequence[float],
    run_greedy: bool,
    verbose: bool = True,
) -> Dict[str, float | None]:
    device = next(model.parameters()).device
    batch = batch.clone().to(device)

    num_loc_val = batch["locs"].shape[1]
    capacity = batch["capacity"] if "capacity" in batch.keys() else torch.ones(batch.batch_size[0], device=device)
    temp_env = CVRPEnv(CVRPGenerator(num_loc=num_loc_val, capacity=capacity.item() if capacity.numel() == 1 else 1.0))
    td = temp_env.reset(batch)
    model.env = temp_env

    results: Dict[str, float | None] = {}

    if run_greedy:
        try:
            out = model(td.clone(), decode_type="greedy", return_actions=True)
            results["greedy"] = -out["reward"].mean().item()
            if verbose:
                print(f"  Greedy: {results['greedy']:.2f}")
        except torch.cuda.OutOfMemoryError:
            results["greedy"] = None
            if verbose:
                print("  Greedy: OOM - skipped")
        torch.cuda.empty_cache()

    for bw in beam_widths:
        key = f"beam_{bw}"
        try:
            out = model(td.clone(), decode_type="beam_search", beam_width=bw, select_best=False, return_actions=True)
            results[key] = -out["reward"].max().item()
            if verbose:
                print(f"  Beam {bw}: {results[key]:.2f}")
        except torch.cuda.OutOfMemoryError:
            results[key] = None
            if verbose:
                print(f"  Beam {bw}: OOM - skipped")
        torch.cuda.empty_cache()

    for temp in beam_temps:
        key = f"beam_{beam_temp_width}_temp{temp}"
        try:
            out = model(
                td.clone(),
                decode_type="beam_search",
                beam_width=beam_temp_width,
                temperature=temp,
                select_best=False,
                return_actions=True,
            )
            results[key] = -out["reward"].max().item()
            if verbose:
                print(f"  Beam {beam_temp_width} (temp={temp}): {results[key]:.2f}")
        except torch.cuda.OutOfMemoryError:
            results[key] = None
            if verbose:
                print(f"  Beam {beam_temp_width} (temp={temp}): OOM - skipped")
        torch.cuda.empty_cache()

    for n in sampling_counts:
        key = f"sampling_{n}"
        try:
            out = model(td.clone(), decode_type="sampling", num_samples=n, return_actions=True)
            results[key] = -out["reward"].max().item()
            if verbose:
                print(f"  Sampling {n}: {results[key]:.2f}")
        except torch.cuda.OutOfMemoryError:
            results[key] = None
            if verbose:
                print(f"  Sampling {n}: OOM - skipped")
        torch.cuda.empty_cache()

    for temp in sampling_temps:
        key = f"sampling_{sampling_temp_count}_temp{temp}"
        try:
            out = model(
                td.clone(),
                decode_type="sampling",
                num_samples=sampling_temp_count,
                temperature=temp,
                return_actions=True,
            )
            results[key] = -out["reward"].max().item()
            if verbose:
                print(f"  Sampling {sampling_temp_count} (temp={temp}): {results[key]:.2f}")
        except torch.cuda.OutOfMemoryError:
            results[key] = None
            if verbose:
                print(f"  Sampling {sampling_temp_count} (temp={temp}): OOM - skipped")
        torch.cuda.empty_cache()

    return results


def method_label(m: str) -> str:
    if m.startswith("beam_"):
        parts = m.split("_")
        width = parts[1]
        temp = parts[2].replace("temp", "") if len(parts) > 2 else "-"
        return f"beam{width}@{temp}"
    if m.startswith("sampling_"):
        parts = m.split("_")
        count = parts[1]
        temp = parts[2].replace("temp", "") if len(parts) > 2 else "-"
        return f"samp{count}@{temp}"
    return m


def summarize_results(results: List[dict], methods: Sequence[str]) -> None:
    print("\n" + "=" * 173)
    print("SUMMARY OF RESULTS")
    print("=" * 173)
    header = (
        f"{'Instance':<20} {'N':<5} {'BKS':<8} {'BKS_norm':<8} "
        + " ".join(f"{method_label(m):<8}" for m in methods)
        + f" {'Best':<10} {'Gap%':<8}"
    )
    print(header)
    print("-" * 190)

    sums = {m: 0.0 for m in methods}
    counts = {m: 0 for m in methods}
    best_sum = 0.0
    best_count = 0
    gap_sum = 0.0
    gap_count = 0

    for r in results:
        vals = [(m, r.get(m)) for m in methods if r.get(m) is not None]
        best_val = min([v for _, v in vals], default=None)
        second_best = None
        if vals:
            greater = [v for _, v in vals if v > best_val]
            second_best = min(greater) if greater else None

        def fmt_mark(method: str) -> str:
            v = r.get(method)
            if v is None:
                return "OOM"
            suffix = ""
            if best_val is not None and v == best_val:
                suffix = "**"
            elif second_best is not None and v == second_best:
                suffix = "*"
            return f"{v:.2f}{suffix}"

        bks_raw = r.get("bks_raw")
        bks_norm = r.get("bks_normalized")
        gap_pct = None
        if bks_norm and best_val is not None:
            gap_pct = 100.0 * (best_val - bks_norm) / bks_norm

        for m, v in vals:
            sums[m] += v
            counts[m] += 1
        if best_val is not None:
            best_sum += best_val
            best_count += 1
        if gap_pct is not None:
            gap_sum += gap_pct
            gap_count += 1

        print(
            f"{r['instance']:<20} {r['num_customers']:<5} "
            f"{(f'{bks_raw:.0f}' if bks_raw else 'N/A'):<8} "
            f"{(f'{bks_norm:.2f}' if bks_norm else 'N/A'):<8} "
            + " ".join(f"{fmt_mark(m):<8}" for m in methods)
            + f" {(f'{best_val:.2f}' if best_val is not None else 'N/A'):<10} "
            f"{(f'{gap_pct:.2f}' if gap_pct is not None else 'N/A'):<8}"
        )

    def mean_fmt(val, cnt):
        return f"{(val / cnt):.2f}" if cnt > 0 else "N/A"

    mean_cells = [mean_fmt(sums[m], counts[m]) for m in methods]
    mean_best = mean_fmt(best_sum, best_count)
    mean_gap = mean_fmt(gap_sum, gap_count)
    print("-" * 190)
    print(
        f"{'MEAN':<20} {'':<5} {'':<8} {'':<8} "
        + " ".join(f"{c:<8}" for c in mean_cells)
        + f" {mean_best:<10} {mean_gap:<8}"
    )


def save_csv(results: List[dict], methods: Sequence[str], csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["Instance", "N", "BKS", "BKS_norm"] + [m for m in methods] + ["Best", "Gap_percent"]
    with csv_path.open("w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            vals = [r[m] for m in methods if r.get(m) is not None]
            best = min(vals) if vals else None
            gap_pct = None
            if r.get("bks_normalized") and best is not None:
                gap_pct = 100.0 * (best - r["bks_normalized"]) / r["bks_normalized"]

            row = {
                "Instance": r["instance"],
                "N": r["num_customers"],
                "BKS": r["bks_raw"] if r.get("bks_raw") else "N/A",
                "BKS_norm": f"{r['bks_normalized']:.2f}" if r.get("bks_normalized") else "N/A",
                "Best": f"{best:.2f}" if best is not None else "N/A",
                "Gap_percent": f"{gap_pct:.2f}" if gap_pct is not None else "N/A",
            }
            for m in methods:
                row[m] = f"{r[m]:.2f}" if r.get(m) is not None else "OOM"
            writer.writerow(row)
    print(f"\nCSV saved to: {csv_path}")


def plot_heatmap(results: List[dict], methods: Sequence[str], out_path: Path) -> None:
    def gap_to_color(gap: float | None):
        if gap is None:
            return "#e0e0e0"
        if gap <= 3.0:
            return "#0b6623"
        if gap >= 20.0:
            return "#ffffff"
        t = (gap - 3.0) / (20.0 - 3.0)
        start = mcolors.to_rgb("#0b6623")
        end = mcolors.to_rgb("#ffffff")
        return tuple(s + t * (e - s) for s, e in zip(start, end))

    cell_text = []
    cell_colors = []
    row_labels = []
    for r in results:
        row_labels.append(r["instance"])
        bks = r.get("bks_normalized")
        vals = []
        cols = []
        for m in methods:
            v = r.get(m)
            if v is None or bks is None:
                vals.append("OOM" if v is None else "N/A")
                cols.append(gap_to_color(None))
            else:
                gap = 100.0 * (v - bks) / bks
                vals.append(f"{v:.2f}\n({gap:.1f}%)")
                cols.append(gap_to_color(gap))
        cell_text.append(vals)
        cell_colors.append(cols)

    fig, ax = plt.subplots(figsize=(len(methods) * 0.9 + 4, len(results) * 0.35 + 2))
    ax.axis("off")
    table = ax.table(
        cellText=cell_text,
        cellColours=cell_colors,
        colLabels=[method_label(m) for m in methods],
        rowLabels=row_labels,
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.2, 1.2)

    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_height(cell.get_height() * 1.6)
        else:
            cell.set_height(cell.get_height() * 1.15)

    cmap = mcolors.LinearSegmentedColormap.from_list("gapgreen", ["#0b6623", "#ffffff"])
    norm = mcolors.Normalize(vmin=3, vmax=20, clip=True)
    cax = fig.add_axes([0.2, 0.04, 0.6, 0.03])
    cb = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation="horizontal")
    cb.set_label("Optimality Gap (%)  (≤3% dark green, 20% white)", fontsize=8)
    cb.ax.tick_params(labelsize=7)

    plt.tight_layout(rect=[0, 0.08, 1, 1])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Heatmap saved to: {out_path}")


# ----------------------------- Entrypoint ----------------------------- #


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CVRP inference (beam search / sampling) runner")
    parser.add_argument("--checkpoint", type=Path, help="Path to model checkpoint")
    parser.add_argument("--load_csv", type=Path, help="Load existing CSV instead of running inference")
    parser.add_argument(
        "--instances",
        type=str,
        nargs="*",
        default=[
            "X-n110-k13",
            "X-n115-k10",
            "X-n153-k22",
            "X-n172-k51",
            "X-n120-k6",
            "X-n129-k18",
            "X-n125-k30",
            "X-n134-k13",
            "X-n344-k43",
            "X-n351-k40",
            "X-n393-k38",
            "X-n420-k130",
            "X-n376-k94",
            "X-n384-k52",
            "X-n317-k53",
            "X-n367-k17",
        ],
        help="List of CVRPLib instance stems. If omitted or --all_instances is set, run all .npz in val_base.",
    )
    parser.add_argument("--all_instances", action="store_true", help="Run all instances found under val_base (ignored if --instances provided)")
    parser.add_argument("--val_base", type=Path, default=Path("cvrplib_instances/cvrplib_x_npz/instances"))
    parser.add_argument("--beam_widths", type=_list_ints, default=_list_ints("10,100,1000,5000"), help="Beam widths")
    parser.add_argument("--beam_temps", type=_list_floats, default=_list_floats("1.5,2.0,3.0"), help="Beam temps")
    parser.add_argument("--beam_temp_width", type=int, default=5000, help="Beam width used with beam temps")
    parser.add_argument("--sampling_counts", type=_list_ints, default=_list_ints("1000,10000,100000"), help="Sampling counts")
    parser.add_argument("--sampling_temp_count", type=int, default=None, help="Sampling count to pair with all sampling temps")
    parser.add_argument("--sampling_temps", type=_list_floats, default=_list_floats("1.0"), help="Sampling temps")
    parser.add_argument("--no_greedy", action="store_true", help="Skip greedy decode")
    parser.add_argument("--csv_path", type=Path, default=None, help="Optional override for CSV output path")
    parser.add_argument("--plot_path", type=Path, default=None, help="Optional override for heatmap PNG path")
    parser.add_argument("--print_table", action="store_true", help="Print summary table to terminal")
    parser.add_argument("--no_plot", action="store_true", help="Skip heatmap plot")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)

    def resolve_out_dir() -> Path:
        if args.load_csv:
            return args.load_csv.parent
        if args.checkpoint is None:
            raise ValueError("Either --checkpoint or --load_csv must be provided")
        parts = list(args.checkpoint.resolve().parts)
        if "checkpoints" in parts:
            idx = parts.index("checkpoints")
            return Path(os.path.join(*parts[:idx]))
        return args.checkpoint.parent

    out_dir = resolve_out_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.csv_path or out_dir / "beam_search_comparison_results.csv"
    plot_path = args.plot_path or out_dir / "beam_search_comparison_heatmap.png"

    if args.load_csv:
        with args.load_csv.open() as f:
            reader = csv.DictReader(f)
            raw_rows = list(reader)
        if not raw_rows:
            raise ValueError(f"No rows found in CSV: {args.load_csv}")
        meta_cols = {"Instance", "N", "BKS", "BKS_norm", "Best", "Gap_percent"}
        method_cols = [c for c in raw_rows[0].keys() if c not in meta_cols]
        results = []
        for row in raw_rows:
            r = {
                "instance": row["Instance"],
                "num_customers": int(row["N"]),
                "bks_raw": float(row["BKS"]) if row["BKS"] not in ("N/A", "") else None,
                "bks_normalized": float(row["BKS_norm"]) if row["BKS_norm"] not in ("N/A", "") else None,
            }
            for m in method_cols:
                val = row.get(m, "OOM")
                if val == "OOM":
                    r[m] = None
                else:
                    try:
                        r[m] = float(val)
                    except ValueError:
                        r[m] = None
            results.append(r)
        methods_order = method_cols
    else:
        if args.checkpoint is None:
            raise ValueError("Either --checkpoint or --load_csv must be provided")

        model = load_model(args.checkpoint, device)
        if args.instances:
            instance_list = args.instances
        elif args.all_instances:
            instance_list = discover_all_instances(args.val_base)
        else:
            raise ValueError("Provide --instances or use --all_instances to select what to run.")

        val_tds = load_instances(instance_list, args.val_base)

        methods_order = []
        if not args.no_greedy:
            methods_order.append("greedy")
        methods_order += [f"beam_{bw}" for bw in args.beam_widths]
        methods_order += [f"beam_{args.beam_temp_width}_temp{t}" for t in args.beam_temps]
        methods_order += [f"sampling_{n}" for n in args.sampling_counts]
        sampling_temp_count = args.sampling_temp_count or (args.sampling_counts[0] if args.sampling_counts else 0)
        methods_order += [f"sampling_{sampling_temp_count}_temp{t}" for t in args.sampling_temps]

        results = []
        print("\nRunning inference with configured decoding strategies...")
        with torch.inference_mode():
            for idx, td in enumerate(val_tds):
                instance_name = instance_list[idx]
                print(f"\nProcessing: {instance_name}")

                bks_raw = load_bks_cost(instance_name)
                bks_norm = calculate_normalized_bks(instance_name)

                inst_res = {
                    "instance": instance_name,
                    "num_customers": td["locs"].shape[1],
                    "bks_raw": bks_raw,
                    "bks_normalized": bks_norm,
                }
                inst_res.update(
                    run_methods_on_instance(
                        model=model,
                        batch=td,
                        beam_widths=args.beam_widths,
                        beam_temps=args.beam_temps,
                        beam_temp_width=args.beam_temp_width,
                        sampling_counts=args.sampling_counts,
                        sampling_temp_count=sampling_temp_count,
                        sampling_temps=args.sampling_temps,
                        run_greedy=not args.no_greedy,
                        verbose=True,
                    )
                )
                results.append(inst_res)

        save_csv(results, methods_order, csv_path)

    if args.print_table:
        summarize_results(results, methods_order)

    if not args.no_plot:
        plot_heatmap(results, methods_order, plot_path)


if __name__ == "__main__":
    main()
