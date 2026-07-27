from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib import font_manager
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent
TIMES_NEW_ROMAN_PATH = Path("/mnt/c/Windows/Fonts/times.ttf")
if TIMES_NEW_ROMAN_PATH.exists():
    font_manager.fontManager.addfont(str(TIMES_NEW_ROMAN_PATH))

rcParams["font.family"] = "serif"
rcParams["font.serif"] = ["Times New Roman", "DejaVu Serif"]
rcParams["mathtext.fontset"] = "custom"
rcParams["mathtext.rm"] = "Times New Roman"
rcParams["mathtext.it"] = "Times New Roman:italic"
rcParams["mathtext.bf"] = "Times New Roman:bold"
rcParams["axes.titlesize"] = 22
rcParams["axes.labelsize"] = 20
rcParams["xtick.labelsize"] = 16
rcParams["ytick.labelsize"] = 16
rcParams["legend.fontsize"] = 16

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data_process.compare_glory_analysis_reanalysis import (  # noqa: E402
    REANALYSIS_ROOT,
    build_date_index,
    coord_name,
    normalize_path,
    open_dataset,
)

DEG_C = "\N{DEGREE SIGN}C"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Fig. 4 depth-wise error profiles.")
    parser.add_argument(
        "--depth_profile_csv",
        type=Path,
        default=REPO_ROOT
        / "outputs"
        / "real_init_error_eval_20220601_20231231_input_norm"
        / "model_compare_depth_profile_metrics_thetao_lead7.csv",
    )
    parser.add_argument("--reanalysis_root", type=Path, default=REANALYSIS_ROOT)
    parser.add_argument("--variable", default="thetao")
    parser.add_argument("--input_kind", default="analysis_init")
    parser.add_argument("--before_name", default="Standard Swin-T")
    parser.add_argument("--after_name", default="FCR (Ours)")
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT
        / "sn-fcr_bundle_20260707"
        / "latex"
        / "figures"
        / "analysis_init_mean_depth_profile.pdf",
    )
    args = parser.parse_args()
    args.depth_profile_csv = normalize_path(args.depth_profile_csv)
    args.reanalysis_root = normalize_path(args.reanalysis_root)
    args.output = normalize_path(args.output)
    return args


def read_depths_m(reanalysis_root: Path, variable: str, n_depths: int) -> np.ndarray:
    by_date = build_date_index(reanalysis_root)
    if not by_date:
        raise FileNotFoundError(f"No reanalysis files found under {reanalysis_root}")
    first_path = by_date[sorted(by_date)[0]]
    with open_dataset(first_path) as ds:
        data = ds[variable]
        depth_name = coord_name(data, ("depth", "elevation"))
        if depth_name is None:
            raise KeyError(f"No depth coordinate found in {first_path}")
        depths = np.asarray(data[depth_name].values[:n_depths], dtype=float)
    if depths.size != n_depths:
        raise ValueError(f"Expected {n_depths} depth levels, found {depths.size}")
    return depths


def load_mean_profiles(
    csv_path: Path,
    input_kind: str,
    model_names: tuple[str, str],
) -> tuple[np.ndarray, dict[str, dict[str, np.ndarray]], int]:
    grouped: dict[str, dict[int, dict[str, list[float]]]] = {
        model_name: defaultdict(lambda: {"rmse": [], "mae": [], "bias": []})
        for model_name in model_names
    }
    dates: set[str] = set()
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["input_kind"] != input_kind or row["model"] not in model_names:
                continue
            model_name = row["model"]
            depth_index = int(row["depth_index"])
            grouped[model_name][depth_index]["rmse"].append(float(row["rmse"]))
            grouped[model_name][depth_index]["mae"].append(float(row["mae"]))
            grouped[model_name][depth_index]["bias"].append(float(row["bias_pred_minus_truth"]))
            dates.add(row["input_date"])

    all_depths = sorted({depth for model_data in grouped.values() for depth in model_data})
    if not all_depths:
        raise ValueError(f"No rows found for input_kind={input_kind!r} and models={model_names}")

    depth_indices = np.asarray(all_depths, dtype=int)
    profiles: dict[str, dict[str, np.ndarray]] = {}
    for model_name in model_names:
        profiles[model_name] = {}
        for metric in ("rmse", "mae", "bias"):
            profiles[model_name][metric] = np.asarray(
                [np.nanmean(grouped[model_name][depth][metric]) for depth in all_depths],
                dtype=float,
            )
    return depth_indices, profiles, len(dates)


def plot_depth_profiles(
    output_path: Path,
    depths_m: np.ndarray,
    profiles: dict[str, dict[str, np.ndarray]],
    before_name: str,
    after_name: str,
) -> None:
    metrics = [
        ("(a) Analysis-initialized RMSE", "rmse", f"RMSE ({DEG_C})"),
        ("(b) Analysis-initialized MAE", "mae", f"MAE ({DEG_C})"),
        ("(c) Analysis-initialized Bias", "bias", f"Bias ({DEG_C})"),
    ]
    colors = {before_name: "#4C78A8", after_name: "#D55E00"}
    linestyles = {before_name: "-", after_name: "--"}

    fig, axes = plt.subplots(1, 3, figsize=(15, 6), constrained_layout=True, sharey=True)
    for ax, (title, metric, xlabel) in zip(axes, metrics):
        for model_name in (before_name, after_name):
            ax.plot(
                profiles[model_name][metric],
                depths_m,
                label=model_name,
                color=colors[model_name],
                linestyle=linestyles[model_name],
                linewidth=2.4,
            )
        if metric == "bias":
            ax.axvline(0.0, color="0.35", linewidth=1.2, linestyle=":")
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.grid(True, color="0.88", linewidth=0.8)
        ax.tick_params(labelsize=16)

    axes[0].set_ylabel("Depth (m)")
    axes[0].invert_yaxis()
    axes[0].legend(frameon=False, loc="lower right")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    if output_path.suffix.lower() != ".png":
        fig.savefig(output_path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    model_names = (args.before_name, args.after_name)
    depth_indices, profiles, n_dates = load_mean_profiles(
        args.depth_profile_csv, args.input_kind, model_names
    )
    depths_m = read_depths_m(args.reanalysis_root, args.variable, int(depth_indices.max()) + 1)[depth_indices]
    plot_depth_profiles(args.output, depths_m, profiles, args.before_name, args.after_name)
    print(f"Saved Fig. 4: {args.output}")
    print(f"Dates averaged: {n_dates}")
    print(f"Depth range: {depths_m[0]:.3f} to {depths_m[-1]:.3f} m")


if __name__ == "__main__":
    main()
