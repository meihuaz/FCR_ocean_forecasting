from __future__ import annotations

import argparse
import csv
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import torch
import torch.nn.functional as F
import yaml

TIMES_NEW_ROMAN_PATH = Path("/mnt/c/Windows/Fonts/times.ttf")
if TIMES_NEW_ROMAN_PATH.exists():
    font_manager.fontManager.addfont(str(TIMES_NEW_ROMAN_PATH))

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "custom",
        "mathtext.rm": "Times New Roman",
        "mathtext.it": "Times New Roman:italic",
        "mathtext.bf": "Times New Roman:bold",
        "axes.titlesize": 22,
        "axes.labelsize": 20,
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
        "legend.fontsize": 16,
    }
)

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data_process.compare_glory_analysis_reanalysis import (
    ANALYSIS_ROOT,
    OUTPUT_DIR,
    REANALYSIS_ROOT,
    build_date_index,
    coord_name,
    normalize_path,
    open_dataset,
    select_range,
)
from model.model_factory import build_model, describe_model_type, extract_prediction


DEFAULT_BEFORE_CKPT = REPO_ROOT / "ckpt" / "swinunet" / "best_ckpt.tar"
DEFAULT_AFTER_CKPT = (
    REPO_ROOT
    / "outputs"
    / "run_en4_bg_model3_pre_pgdTrue_simsiam_eps0.0005_lambda0.6"
    / "training_checkpoints"
    / "ckpt_51.tar"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Use GLORY analysis fields and optional same-date reanalysis fields as initial "
            "conditions, use future GLORY reanalysis fields as truth, and compare two "
            "checkpoints before/after SimSiam training."
        )
    )
    parser.add_argument("--yaml_config", default=REPO_ROOT / "config" / "en4_Transformer_bg_model3.yaml", type=Path)
    parser.add_argument("--analysis_root", default=ANALYSIS_ROOT, type=Path)
    parser.add_argument("--reanalysis_root", default=REANALYSIS_ROOT, type=Path)
    parser.add_argument("--output_dir", default=OUTPUT_DIR, type=Path)
    parser.add_argument("--before_ckpt", default=DEFAULT_BEFORE_CKPT, type=Path)
    parser.add_argument("--after_ckpt", default=DEFAULT_AFTER_CKPT, type=Path)
    parser.add_argument("--before_name", default="Standard Swin-T")
    parser.add_argument("--after_name", default="FCR (Ours)")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--variable", default="thetao")
    parser.add_argument(
        "--lead_days",
        default=7,
        type=int,
        help="Prediction lead time. Input is analysis(date); truth is reanalysis(date + lead_days).",
    )
    parser.add_argument("--date_start", default=None, help="Inclusive input analysis date, e.g. 2022-06-01.")
    parser.add_argument("--date_end", default=None, help="Inclusive input analysis date, e.g. 2022-06-12.")
    parser.add_argument("--limit", default=None, type=int)
    parser.add_argument("--lon_min", default=100.0, type=float)
    parser.add_argument("--lon_max", default=180.0, type=float)
    parser.add_argument("--lat_min", default=0.0, type=float)
    parser.add_argument("--lat_max", default=60.0, type=float)
    parser.add_argument(
        "--depth_levels",
        default=None,
        type=int,
        help="Number of leading depth levels. Defaults to img_size[2] from yaml.",
    )
    parser.add_argument(
        "--output_csv",
        default=None,
        type=Path,
        help="Output CSV path. Defaults to output_dir/model_compare_analysis_init_reanalysis_truth_<variable>.csv.",
    )
    parser.add_argument(
        "--degradation_csv",
        default=None,
        type=Path,
        help=(
            "Output CSV for paired near-real initial-error degradation. Defaults to "
            "output_dir/model_compare_real_init_error_degradation_<variable>_lead<lead_days>.csv."
        ),
    )
    parser.add_argument(
        "--depth_profile_csv",
        default=None,
        type=Path,
        help=(
            "Intermediate CSV for per-date, per-depth RMSE/MAE/bias. Defaults to "
            "output_dir/model_compare_depth_profile_metrics_<variable>_lead<lead_days>.csv."
        ),
    )
    parser.add_argument(
        "--plot_from_depth_profile_csv",
        default=None,
        type=Path,
        help="Redraw mean depth-profile figures from a cached depth-profile CSV without rerunning models.",
    )
    parser.add_argument(
        "--compare_reanalysis_init",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Also evaluate same-date reanalysis as a cleaner initial-condition baseline. "
            "This enables near-real analysis-minus-reanalysis initial-error degradation."
        ),
    )
    parser.add_argument(
        "--normalization_reference",
        choices=("input", "reanalysis_init"),
        default="input",
        help=(
            "Field used for min/max normalization. 'input' matches the training/evaluation "
            "pipeline; 'reanalysis_init' keeps analysis and reanalysis initializations on "
            "the same scale for a stricter paired sensitivity test."
        ),
    )
    parser.add_argument(
        "--plot",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save prediction/truth/error figures.",
    )
    parser.add_argument(
        "--plot_limit",
        default=4,
        type=int,
        help="Only save per-date depth/spatial figures for the first N dates. Use -1 for all dates.",
    )
    parser.add_argument(
        "--plot_depth_index",
        default=0,
        type=int,
        help="Depth index used for spatial maps.",
    )
    parser.add_argument(
        "--plot_dir",
        default=None,
        type=Path,
        help="Directory for figures. Defaults to <output_dir>/figures/model_compare_analysis_init_reanalysis_truth.",
    )
    parser.add_argument(
        "--fig3_case_output",
        default=None,
        type=Path,
        help="Optional output path for the 3x3 Fig. 3 case layout.",
    )
    args = parser.parse_args()

    args.yaml_config = normalize_path(args.yaml_config)
    args.analysis_root = normalize_path(args.analysis_root)
    args.reanalysis_root = normalize_path(args.reanalysis_root)
    args.output_dir = normalize_path(args.output_dir)
    args.before_ckpt = normalize_path(args.before_ckpt)
    args.after_ckpt = normalize_path(args.after_ckpt)
    if args.output_csv is not None:
        args.output_csv = normalize_path(args.output_csv)
    if args.degradation_csv is not None:
        args.degradation_csv = normalize_path(args.degradation_csv)
    if args.depth_profile_csv is not None:
        args.depth_profile_csv = normalize_path(args.depth_profile_csv)
    if args.plot_from_depth_profile_csv is not None:
        args.plot_from_depth_profile_csv = normalize_path(args.plot_from_depth_profile_csv)
    if args.plot_dir is not None:
        args.plot_dir = normalize_path(args.plot_dir)
    if args.fig3_case_output is not None:
        args.fig3_case_output = normalize_path(args.fig3_case_output)
    return args


def resolve_device(device_text: str) -> torch.device:
    if not torch.cuda.is_available() or not device_text.startswith("cuda"):
        return torch.device("cpu" if device_text.startswith("cuda") else device_text)

    if device_text == "cuda":
        return torch.device("cuda:0")

    try:
        requested_index = int(device_text.split(":")[1])
    except (IndexError, ValueError):
        return torch.device("cuda:0")

    if requested_index >= torch.cuda.device_count():
        return torch.device("cuda:0")
    return torch.device(device_text)


def load_params(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        params = yaml.safe_load(f)
    params["log_file"] = None
    return params


def load_model(params: dict, checkpoint_path: Path, device: torch.device) -> torch.nn.Module:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    model = build_model(params).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint["model_state"] if isinstance(checkpoint, dict) and "model_state" in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model.eval()
    return model


def date_shift(date_text: str, days: int) -> str:
    return (datetime.strptime(date_text, "%Y-%m-%d") + timedelta(days=days)).strftime("%Y-%m-%d")


def select_model_field(dataset_path: Path, variable: str, args: argparse.Namespace, depth_levels: int) -> np.ndarray:
    with open_dataset(dataset_path) as ds:
        if variable not in ds:
            raise KeyError(f"{variable!r} not found in {dataset_path}. Available variables: {list(ds.data_vars)}")

        data = ds[variable]

        time_name = coord_name(data, ("time",))
        if time_name is not None and time_name in data.dims and data.sizes[time_name] == 1:
            data = data.isel({time_name: 0})

        depth_name = coord_name(data, ("depth", "elevation"))
        if depth_name is None or depth_name not in data.dims:
            raise KeyError(f"No depth coordinate found in {dataset_path}")
        data = data.isel({depth_name: slice(0, depth_levels)})

        lat_name = coord_name(data, ("latitude", "lat"))
        lon_name = coord_name(data, ("longitude", "lon"))
        if lat_name is not None:
            data = select_range(data, lat_name, args.lat_min, args.lat_max)
        if lon_name is not None:
            data = select_range(data, lon_name, args.lon_min, args.lon_max)

        data = data.transpose(lat_name, lon_name, depth_name)
        return np.asarray(data.values, dtype=np.float32)


def normalize_with_initial_field(
    input_field: np.ndarray,
    truth_field: np.ndarray,
    scale_field: np.ndarray | None = None,
    valid_fields: list[np.ndarray] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    valid_mask = np.isfinite(input_field) & np.isfinite(truth_field)
    if valid_fields is not None:
        for field in valid_fields:
            valid_mask &= np.isfinite(field)
    if not valid_mask.any():
        raise ValueError("Input/truth fields have no overlapping finite ocean points.")

    scale_source = input_field if scale_field is None else scale_field
    min_value = float(np.nanmin(scale_source))
    max_value = float(np.nanmax(scale_source))
    scale = max_value - min_value
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Input field has invalid min/max; cannot normalize.")

    input_norm = (input_field - min_value) / scale
    truth_norm = (truth_field - min_value) / scale
    input_norm = np.nan_to_num(input_norm, nan=0.0)
    truth_norm = np.nan_to_num(truth_norm, nan=0.0)
    return input_norm, truth_norm, valid_mask, min_value, max_value


def prepare_tensors(
    input_norm: np.ndarray,
    truth_norm: np.ndarray,
    valid_mask: np.ndarray,
    params: dict,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, np.ndarray]:
    height, width, depth = params["img_size"]

    input_tensor = torch.from_numpy(input_norm[None]).float().to(device)
    input_tensor = input_tensor.permute(0, 3, 1, 2)
    input_tensor = F.interpolate(input_tensor, size=(height, width), mode="nearest")

    truth_tensor = torch.from_numpy(truth_norm[None]).float().to(device)
    truth_tensor = truth_tensor.permute(0, 3, 1, 2)
    truth_tensor = F.interpolate(truth_tensor, size=(height, width), mode="nearest")
    truth_tensor = truth_tensor.permute(0, 2, 3, 1)

    mask_tensor = torch.from_numpy(valid_mask[None].astype(np.float32)).to(device)
    mask_tensor = mask_tensor.permute(0, 3, 1, 2)
    mask_tensor = F.interpolate(mask_tensor, size=(height, width), mode="nearest")
    mask_tensor = mask_tensor.permute(0, 2, 3, 1)
    resized_valid_mask = mask_tensor.detach().cpu().numpy()[0] > 0.5

    if input_tensor.shape[1] != depth or truth_tensor.shape[-1] != depth:
        raise ValueError(
            f"Depth mismatch after preprocessing: input {tuple(input_tensor.shape)}, "
            f"truth {tuple(truth_tensor.shape)}, expected depth {depth}"
        )
    return input_tensor, truth_tensor, resized_valid_mask


def metrics(prediction: np.ndarray, truth: np.ndarray) -> dict[str, float | int]:
    mask = np.isfinite(prediction) & np.isfinite(truth)
    valid_count = int(mask.sum())
    total_count = int(mask.size)
    if valid_count == 0:
        return {
            "total_count": total_count,
            "valid_count": 0,
            "valid_fraction": 0.0,
            "bias_pred_minus_truth": np.nan,
            "mae": np.nan,
            "rmse": np.nan,
            "median_abs_diff": np.nan,
            "p95_abs_diff": np.nan,
            "max_abs_diff": np.nan,
            "corr": np.nan,
        }

    pred = prediction[mask]
    target = truth[mask]
    diff = pred - target
    abs_diff = np.abs(diff)
    if valid_count > 1 and np.nanstd(pred) > 0 and np.nanstd(target) > 0:
        corr = float(np.corrcoef(pred, target)[0, 1])
    else:
        corr = np.nan

    return {
        "total_count": total_count,
        "valid_count": valid_count,
        "valid_fraction": valid_count / total_count,
        "bias_pred_minus_truth": float(np.nanmean(diff)),
        "mae": float(np.nanmean(abs_diff)),
        "rmse": float(np.sqrt(np.nanmean(diff * diff))),
        "median_abs_diff": float(np.nanmedian(abs_diff)),
        "p95_abs_diff": float(np.nanpercentile(abs_diff, 95)),
        "max_abs_diff": float(np.nanmax(abs_diff)),
        "corr": corr,
    }


def evaluate_model(
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
    truth_tensor: torch.Tensor,
    valid_mask: np.ndarray,
    min_value: float,
    max_value: float,
) -> tuple[dict[str, float | int], np.ndarray, np.ndarray]:
    scale = max_value - min_value
    with torch.no_grad():
        prediction = extract_prediction(model(input_tensor))
        prediction = prediction.permute(0, 2, 3, 1)

    prediction_physical = prediction.detach().cpu().numpy()[0] * scale + min_value
    truth_physical = truth_tensor.detach().cpu().numpy()[0] * scale + min_value
    prediction_physical = np.where(valid_mask, prediction_physical, np.nan)
    truth_physical = np.where(valid_mask, truth_physical, np.nan)
    return metrics(prediction_physical, truth_physical), prediction_physical, truth_physical


def field_color_limits(*arrays: np.ndarray, percentile: float = 1.0) -> tuple[float, float]:
    finite_parts = [array[np.isfinite(array)].reshape(-1) for array in arrays if np.isfinite(array).any()]
    if not finite_parts:
        return 0.0, 1.0
    values = np.concatenate(finite_parts)
    vmin, vmax = np.nanpercentile(values, [percentile, 100.0 - percentile])
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
        vmin = float(np.nanmin(values))
        vmax = float(np.nanmax(values))
    if vmin == vmax:
        margin = max(abs(vmin) * 0.01, 1.0)
        return float(vmin - margin), float(vmax + margin)
    return float(vmin), float(vmax)


def symmetric_limit(*arrays: np.ndarray, percentile: float = 99.0) -> float:
    finite_parts = [np.abs(array[np.isfinite(array)]).reshape(-1) for array in arrays if np.isfinite(array).any()]
    if not finite_parts:
        return 1.0
    values = np.concatenate(finite_parts)
    limit = float(np.nanpercentile(values, percentile))
    if not np.isfinite(limit) or limit <= 0:
        limit = float(np.nanmax(values)) if values.size else 1.0
    return limit if limit > 0 else 1.0


def depth_metrics_by_level(prediction: np.ndarray, truth: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    diff = prediction - truth
    mae = np.nanmean(np.abs(diff), axis=(0, 1))
    rmse = np.sqrt(np.nanmean(diff * diff, axis=(0, 1)))
    bias = np.nanmean(diff, axis=(0, 1))
    return mae, rmse, bias


def plot_spatial_comparison(
    output_path: Path,
    input_date: str,
    truth_date: str,
    depth_index: int,
    before_name: str,
    after_name: str,
    truth: np.ndarray,
    before_pred: np.ndarray,
    after_pred: np.ndarray,
) -> None:
    if depth_index < 0 or depth_index >= truth.shape[-1]:
        raise ValueError(f"plot_depth_index={depth_index} is out of range for depth={truth.shape[-1]}")

    truth_2d = truth[:, :, depth_index]
    before_2d = before_pred[:, :, depth_index]
    after_2d = after_pred[:, :, depth_index]
    before_err = before_2d - truth_2d
    after_err = after_2d - truth_2d
    improvement = np.abs(before_err) - np.abs(after_err)

    vmin, vmax = field_color_limits(truth_2d, before_2d, after_2d)
    err_limit = symmetric_limit(before_err, after_err)
    improve_limit = symmetric_limit(improvement)

    fig, axes = plt.subplots(2, 3, figsize=(15, 8.2), constrained_layout=True)
    panels = [
        ("Truth reanalysis", truth_2d, "viridis", vmin, vmax),
        (f"Prediction {before_name}", before_2d, "viridis", vmin, vmax),
        (f"Prediction {after_name}", after_2d, "viridis", vmin, vmax),
        (f"Error {before_name}", before_err, "RdBu_r", -err_limit, err_limit),
        (f"Error {after_name}", after_err, "RdBu_r", -err_limit, err_limit),
        ("|Standard error| - |FCR error|", improvement, "RdBu_r", -improve_limit, improve_limit),
    ]
    for ax, (title, data, cmap, panel_vmin, panel_vmax) in zip(axes.ravel(), panels):
        image = ax.imshow(
            data,
            origin="lower",
            cmap=cmap,
            vmin=panel_vmin,
            vmax=panel_vmax,
            aspect="auto",
        )
        ax.set_title(title)
        ax.set_xlabel("x index")
        ax.set_ylabel("y index")
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(f"analysis {input_date} -> reanalysis {truth_date}, depth index {depth_index}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=170)
    plt.close(fig)


def format_lon_tick(value: float) -> str:
    return f"{int(value)}$^{{\\circ}}$E"


def format_lat_tick(value: float) -> str:
    if value == 0:
        return "0$^{\\circ}$"
    return f"{int(value)}$^{{\\circ}}$N"


def plot_fig3_case_layout(
    output_path: Path,
    input_date: str,
    truth_date: str,
    depth_index: int,
    before_name: str,
    after_name: str,
    reanalysis_initial: np.ndarray,
    analysis_initial: np.ndarray,
    reanalysis_truth: np.ndarray,
    analysis_truth: np.ndarray,
    reanalysis_predictions: dict[str, np.ndarray],
    analysis_predictions: dict[str, np.ndarray],
    lon_min: float,
    lon_max: float,
    lat_min: float,
    lat_max: float,
) -> None:
    if depth_index < 0 or depth_index >= reanalysis_initial.shape[-1]:
        raise ValueError(
            f"plot_depth_index={depth_index} is out of range for depth={reanalysis_initial.shape[-1]}"
        )

    rean_init = reanalysis_initial[:, :, depth_index]
    ana_init = analysis_initial[:, :, depth_index]
    init_diff = ana_init - rean_init

    rean_truth_2d = reanalysis_truth[:, :, depth_index]
    ana_truth_2d = analysis_truth[:, :, depth_index]
    rean_swin_err = reanalysis_predictions[before_name][:, :, depth_index] - rean_truth_2d
    rean_fcr_err = reanalysis_predictions[after_name][:, :, depth_index] - rean_truth_2d
    ana_swin_err = analysis_predictions[before_name][:, :, depth_index] - ana_truth_2d
    ana_fcr_err = analysis_predictions[after_name][:, :, depth_index] - ana_truth_2d
    rean_reduction = np.abs(rean_swin_err) - np.abs(rean_fcr_err)
    ana_reduction = np.abs(ana_swin_err) - np.abs(ana_fcr_err)

    state_vmin, state_vmax = field_color_limits(rean_init, ana_init)
    init_diff_limit = symmetric_limit(init_diff)
    error_limit = symmetric_limit(rean_swin_err, rean_fcr_err, ana_swin_err, ana_fcr_err)
    reduction_limit = symmetric_limit(rean_reduction, ana_reduction)

    cmap_state = plt.get_cmap("viridis").copy()
    cmap_state.set_bad("white")
    cmap_diff = plt.get_cmap("RdBu_r").copy()
    cmap_diff.set_bad("white")
    extent = (lon_min, lon_max, lat_min, lat_max)
    title_size = 22
    label_size = 20
    tick_size = 16
    cbar_label_size = 18

    fig, axes = plt.subplots(3, 3, figsize=(20, 15), constrained_layout=True)
    panels = [
        ("(a) Reanalysis Initial Field", rean_init, cmap_state, state_vmin, state_vmax, "Temperature (\N{DEGREE SIGN}C)"),
        ("(b) Analysis Initial Field", ana_init, cmap_state, state_vmin, state_vmax, "Temperature (\N{DEGREE SIGN}C)"),
        ("(c) Initial Difference", init_diff, cmap_diff, -init_diff_limit, init_diff_limit, "Analysis - reanalysis (\N{DEGREE SIGN}C)"),
        ("(d) Swin Error, Reanalysis Init.", rean_swin_err, cmap_diff, -error_limit, error_limit, "Forecast error (\N{DEGREE SIGN}C)"),
        ("(e) FCR Error, Reanalysis Init.", rean_fcr_err, cmap_diff, -error_limit, error_limit, "Forecast error (\N{DEGREE SIGN}C)"),
        ("(f) FCR Error Reduction", rean_reduction, cmap_diff, -reduction_limit, reduction_limit, "Error reduction (\N{DEGREE SIGN}C)"),
        ("(g) Swin Error, Analysis Init.", ana_swin_err, cmap_diff, -error_limit, error_limit, "Forecast error (\N{DEGREE SIGN}C)"),
        ("(h) FCR Error, Analysis Init.", ana_fcr_err, cmap_diff, -error_limit, error_limit, "Forecast error (\N{DEGREE SIGN}C)"),
        ("(i) FCR Error Reduction", ana_reduction, cmap_diff, -reduction_limit, reduction_limit, "Error reduction (\N{DEGREE SIGN}C)"),
    ]

    lon_ticks = [100, 120, 140, 160, 180]
    lat_ticks = [0, 15, 30, 45, 60]
    for ax, (title, data, cmap, vmin, vmax, label) in zip(axes.flat, panels):
        image = ax.imshow(
            data,
            origin="lower",
            cmap=cmap,
            extent=extent,
            vmin=vmin,
            vmax=vmax,
        )
        ax.set_title(title, fontsize=title_size)
        ax.set_xlabel("Longitude", fontsize=label_size)
        ax.set_ylabel("Latitude", fontsize=label_size)
        ax.set_xticks(lon_ticks)
        ax.set_yticks(lat_ticks)
        ax.set_xticklabels([format_lon_tick(t) for t in lon_ticks])
        ax.set_yticklabels([format_lat_tick(t) for t in lat_ticks])
        ax.tick_params(labelsize=tick_size)
        cb = fig.colorbar(image, ax=ax, shrink=0.86, pad=0.02, label=label)
        cb.ax.tick_params(labelsize=tick_size)
        cb.set_label(cb.ax.get_ylabel(), fontsize=cbar_label_size)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    if output_path.suffix.lower() != ".png":
        fig.savefig(output_path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_depth_profile(
    output_path: Path,
    input_date: str,
    truth_date: str,
    before_name: str,
    after_name: str,
    truth: np.ndarray,
    before_pred: np.ndarray,
    after_pred: np.ndarray,
) -> None:
    before_mae, before_rmse, before_bias = depth_metrics_by_level(before_pred, truth)
    after_mae, after_rmse, after_bias = depth_metrics_by_level(after_pred, truth)
    depth_index = np.arange(truth.shape[-1])

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2), constrained_layout=True)
    axes[0].plot(depth_index, before_rmse, label=before_name)
    axes[0].plot(depth_index, after_rmse, label=after_name)
    axes[0].set_title("RMSE by depth")
    axes[0].set_xlabel("Depth index")
    axes[0].set_ylabel("RMSE")
    axes[0].legend()

    axes[1].plot(depth_index, before_mae, label=before_name)
    axes[1].plot(depth_index, after_mae, label=after_name)
    axes[1].set_title("MAE by depth")
    axes[1].set_xlabel("Depth index")
    axes[1].set_ylabel("MAE")
    axes[1].legend()

    axes[2].plot(depth_index, before_bias, label=before_name)
    axes[2].plot(depth_index, after_bias, label=after_name)
    axes[2].set_title("Bias by depth")
    axes[2].set_xlabel("Depth index")
    axes[2].set_ylabel("Prediction - truth")
    axes[2].legend()

    fig.suptitle(f"input {input_date} -> reanalysis {truth_date}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=170)
    plt.close(fig)


def summarize(rows: list[dict[str, object]], model_name: str) -> None:
    selected = [row for row in rows if row["model"] == model_name]
    rmse = np.asarray([float(row["rmse"]) for row in selected], dtype=np.float64)
    mae = np.asarray([float(row["mae"]) for row in selected], dtype=np.float64)
    bias = np.asarray([float(row["bias_pred_minus_truth"]) for row in selected], dtype=np.float64)
    corr = np.asarray([float(row["corr"]) for row in selected], dtype=np.float64)
    print(
        f"{model_name}: dates={len(selected)}, "
        f"mean_bias={np.nanmean(bias):.4f}, mean_MAE={np.nanmean(mae):.4f}, "
        f"mean_RMSE={np.nanmean(rmse):.4f}, mean_corr={np.nanmean(corr):.5f}"
    )


def build_degradation_rows(
    rows: list[dict[str, object]],
    clean_input_kind: str = "reanalysis_init",
    perturbed_input_kind: str = "analysis_init",
) -> list[dict[str, object]]:
    by_key = {
        (row["input_date"], row["truth_date"], row["model"], row["input_kind"]): row
        for row in rows
    }
    degradation_rows: list[dict[str, object]] = []

    for row in rows:
        if row["input_kind"] != perturbed_input_kind:
            continue
        clean_key = (row["input_date"], row["truth_date"], row["model"], clean_input_kind)
        clean_row = by_key.get(clean_key)
        if clean_row is None:
            continue

        clean_rmse = float(clean_row["rmse"])
        perturbed_rmse = float(row["rmse"])
        clean_mae = float(clean_row["mae"])
        perturbed_mae = float(row["mae"])
        clean_corr = float(clean_row["corr"])
        perturbed_corr = float(row["corr"])
        degradation_rows.append(
            {
                "input_date": row["input_date"],
                "truth_date": row["truth_date"],
                "lead_days": row["lead_days"],
                "model": row["model"],
                "clean_input_kind": clean_input_kind,
                "perturbed_input_kind": perturbed_input_kind,
                "clean_input_file": clean_row["input_file"],
                "perturbed_input_file": row["input_file"],
                "truth_file": row["truth_file"],
                "clean_rmse": clean_rmse,
                "perturbed_rmse": perturbed_rmse,
                "delta_rmse": perturbed_rmse - clean_rmse,
                "relative_delta_rmse": (perturbed_rmse - clean_rmse) / clean_rmse if clean_rmse != 0 else np.nan,
                "clean_mae": clean_mae,
                "perturbed_mae": perturbed_mae,
                "delta_mae": perturbed_mae - clean_mae,
                "relative_delta_mae": (perturbed_mae - clean_mae) / clean_mae if clean_mae != 0 else np.nan,
                "clean_corr": clean_corr,
                "perturbed_corr": perturbed_corr,
                "delta_corr": perturbed_corr - clean_corr,
            }
        )
    return degradation_rows


def print_degradation_summary(degradation_rows: list[dict[str, object]], before_name: str, after_name: str) -> None:
    if not degradation_rows:
        return

    print()
    print("Near-real initial-error degradation")
    by_model: dict[str, list[dict[str, object]]] = {}
    for row in degradation_rows:
        by_model.setdefault(str(row["model"]), []).append(row)

    for model_name, model_rows in sorted(by_model.items()):
        delta_rmse = np.asarray([float(row["delta_rmse"]) for row in model_rows], dtype=np.float64)
        delta_mae = np.asarray([float(row["delta_mae"]) for row in model_rows], dtype=np.float64)
        delta_corr = np.asarray([float(row["delta_corr"]) for row in model_rows], dtype=np.float64)
        print(
            f"{model_name}: dates={len(model_rows)}, "
            f"mean_analysis_minus_reanalysis_RMSE={np.nanmean(delta_rmse):+.4f}, "
            f"mean_analysis_minus_reanalysis_MAE={np.nanmean(delta_mae):+.4f}, "
            f"mean_delta_corr={np.nanmean(delta_corr):+.5f}"
        )

    before_rows = {
        (row["input_date"], row["truth_date"]): row
        for row in degradation_rows
        if row["model"] == before_name
    }
    after_rows = {
        (row["input_date"], row["truth_date"]): row
        for row in degradation_rows
        if row["model"] == after_name
    }
    shared_keys = sorted(set(before_rows) & set(after_rows))
    if shared_keys:
        rmse_gain = np.asarray(
            [
                float(before_rows[key]["delta_rmse"]) - float(after_rows[key]["delta_rmse"])
                for key in shared_keys
            ],
            dtype=np.float64,
        )
        mae_gain = np.asarray(
            [
                float(before_rows[key]["delta_mae"]) - float(after_rows[key]["delta_mae"])
                for key in shared_keys
            ],
            dtype=np.float64,
        )
        print(
            f"{after_name} robustness gain vs {before_name}: "
            f"mean_delta_RMSE_reduction={np.nanmean(rmse_gain):+.4f}, "
            f"mean_delta_MAE_reduction={np.nanmean(mae_gain):+.4f}"
        )


def plot_mean_depth_profiles_from_csv(
    depth_profile_csv: Path,
    plot_dir: Path,
    before_name: str,
    after_name: str,
) -> list[Path]:
    with depth_profile_csv.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"No rows found in depth-profile CSV: {depth_profile_csv}")

    grouped: dict[str, dict[str, dict[int, dict[str, list[float]]]]] = {}
    dates_by_kind: dict[str, set[str]] = {}
    for row in rows:
        input_kind = row["input_kind"]
        model_name = row["model"]
        depth_index = int(row["depth_index"])
        grouped.setdefault(input_kind, {}).setdefault(model_name, {}).setdefault(
            depth_index,
            {"rmse": [], "mae": [], "bias": []},
        )
        grouped[input_kind][model_name][depth_index]["rmse"].append(float(row["rmse"]))
        grouped[input_kind][model_name][depth_index]["mae"].append(float(row["mae"]))
        grouped[input_kind][model_name][depth_index]["bias"].append(float(row["bias_pred_minus_truth"]))
        dates_by_kind.setdefault(input_kind, set()).add(row["input_date"])

    saved_paths: list[Path] = []
    for input_kind, by_model in sorted(grouped.items()):
        model_order = [name for name in (before_name, after_name) if name in by_model]
        model_order.extend(sorted(name for name in by_model if name not in model_order))
        all_depths = sorted({depth for model_data in by_model.values() for depth in model_data})
        depth_index_array = np.asarray(all_depths, dtype=int)

        fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2), constrained_layout=True)
        for model_name in model_order:
            model_data = by_model[model_name]
            mean_rmse = [np.nanmean(model_data[depth]["rmse"]) for depth in all_depths]
            mean_mae = [np.nanmean(model_data[depth]["mae"]) for depth in all_depths]
            mean_bias = [np.nanmean(model_data[depth]["bias"]) for depth in all_depths]
            axes[0].plot(depth_index_array, mean_rmse, label=model_name)
            axes[1].plot(depth_index_array, mean_mae, label=model_name)
            axes[2].plot(depth_index_array, mean_bias, label=model_name)

        axes[0].set_title("Mean RMSE by depth")
        axes[0].set_xlabel("Depth index")
        axes[0].set_ylabel("RMSE")
        axes[1].set_title("Mean MAE by depth")
        axes[1].set_xlabel("Depth index")
        axes[1].set_ylabel("MAE")
        axes[2].set_title("Mean bias by depth")
        axes[2].set_xlabel("Depth index")
        axes[2].set_ylabel("Prediction - truth")
        for ax in axes:
            ax.legend()
        fig.suptitle(f"{input_kind}: mean over {len(dates_by_kind.get(input_kind, set()))} dates")

        output_path = plot_dir / input_kind / "mean_depth_profile.png"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=170)
        plt.close(fig)
        saved_paths.append(output_path)

    return saved_paths


def main() -> None:
    args = parse_args()
    params = load_params(args.yaml_config)
    depth_levels = args.depth_levels or int(params["img_size"][2])
    device = resolve_device(args.device)
    plot_dir = args.plot_dir or (
        args.output_dir / "figures" / f"model_compare_analysis_init_reanalysis_truth_{args.variable}_lead{args.lead_days}"
    )

    if args.plot_from_depth_profile_csv is not None:
        saved_paths = plot_mean_depth_profiles_from_csv(
            args.plot_from_depth_profile_csv,
            plot_dir,
            args.before_name,
            args.after_name,
        )
        print(f"Redrew {len(saved_paths)} mean depth-profile figure(s) from {args.plot_from_depth_profile_csv}")
        for path in saved_paths:
            print(f"Saved figure: {path}")
        return

    analysis_by_date = build_date_index(args.analysis_root)
    reanalysis_by_date = build_date_index(args.reanalysis_root)

    candidate_dates = []
    for input_date in sorted(analysis_by_date):
        if args.date_start is not None and input_date < args.date_start:
            continue
        if args.date_end is not None and input_date > args.date_end:
            continue
        truth_date = date_shift(input_date, args.lead_days)
        if args.compare_reanalysis_init and input_date not in reanalysis_by_date:
            continue
        if truth_date in reanalysis_by_date:
            candidate_dates.append((input_date, truth_date))
    if args.limit is not None:
        candidate_dates = candidate_dates[: args.limit]
    if not candidate_dates:
        raise FileNotFoundError("No analysis/reanalysis input-truth pairs matched the requested dates.")

    print(f"Using {describe_model_type(params.get('model_type', 'swin'))} model on {device}")
    print(f"Analysis input dates: {len(candidate_dates)}")
    print(f"Truth lead days: {args.lead_days}")
    print(f"Domain: lon {args.lon_min} to {args.lon_max}, lat {args.lat_min} to {args.lat_max}")
    print(f"Depth levels: {depth_levels}")
    print(f"Compare same-date reanalysis initial baseline: {args.compare_reanalysis_init}")
    print(f"Normalization reference: {args.normalization_reference}")
    print(f"Before checkpoint: {args.before_ckpt}")
    print(f"After checkpoint: {args.after_ckpt}")
    print()

    before_model = load_model(params, args.before_ckpt, device)
    after_model = load_model(params, args.after_ckpt, device)

    rows: list[dict[str, object]] = []
    input_kind_names = ["reanalysis_init", "analysis_init"] if args.compare_reanalysis_init else ["analysis_init"]
    depth_profile_accumulator: dict[str, dict[str, dict[str, list[np.ndarray]]]] = {
        input_kind: {
            args.before_name: {"mae": [], "rmse": [], "bias": []},
            args.after_name: {"mae": [], "rmse": [], "bias": []},
        }
        for input_kind in input_kind_names
    }
    saved_figures: list[Path] = []
    depth_profile_csv = args.depth_profile_csv or (
        args.output_dir / f"model_compare_depth_profile_metrics_{args.variable}_lead{args.lead_days}.csv"
    )
    depth_profile_fieldnames = [
        "input_date",
        "truth_date",
        "lead_days",
        "input_kind",
        "normalization_reference",
        "model",
        "checkpoint",
        "input_file",
        "truth_file",
        "depth_index",
        "mae",
        "rmse",
        "bias_pred_minus_truth",
    ]
    depth_profile_csv.parent.mkdir(parents=True, exist_ok=True)
    depth_profile_file = depth_profile_csv.open("w", newline="", encoding="utf-8")
    depth_profile_writer = csv.DictWriter(depth_profile_file, fieldnames=depth_profile_fieldnames)
    depth_profile_writer.writeheader()

    for date_index, (input_date, truth_date) in enumerate(candidate_dates):
        analysis_input_field = select_model_field(analysis_by_date[input_date], args.variable, args, depth_levels)
        reanalysis_input_field = (
            select_model_field(reanalysis_by_date[input_date], args.variable, args, depth_levels)
            if args.compare_reanalysis_init
            else None
        )
        truth_field = select_model_field(reanalysis_by_date[truth_date], args.variable, args, depth_levels)

        if reanalysis_input_field is not None:
            shared_valid_fields = [analysis_input_field, reanalysis_input_field, truth_field]
            input_specs = [
                ("reanalysis_init", reanalysis_input_field, reanalysis_by_date[input_date]),
                ("analysis_init", analysis_input_field, analysis_by_date[input_date]),
            ]
        else:
            shared_valid_fields = [analysis_input_field, truth_field]
            input_specs = [("analysis_init", analysis_input_field, analysis_by_date[input_date])]

        model_specs = [
            (args.before_name, before_model, args.before_ckpt),
            (args.after_name, after_model, args.after_ckpt),
        ]
        date_results_by_input: dict[str, list[dict[str, object]]] = {}
        fig3_payload: dict[str, dict[str, object]] = {}
        for input_kind, input_field, input_path in input_specs:
            scale_field = (
                reanalysis_input_field
                if args.normalization_reference == "reanalysis_init" and reanalysis_input_field is not None
                else input_field
            )
            input_norm, truth_norm, valid_mask, min_value, max_value = normalize_with_initial_field(
                input_field,
                truth_field,
                scale_field=scale_field,
                valid_fields=shared_valid_fields,
            )
            input_tensor, truth_tensor, resized_valid_mask = prepare_tensors(
                input_norm,
                truth_norm,
                valid_mask,
                params,
                device,
            )
            scale = max_value - min_value
            input_physical = input_tensor.permute(0, 2, 3, 1).detach().cpu().numpy()[0] * scale + min_value
            input_physical = np.where(resized_valid_mask, input_physical, np.nan)

            date_results = []
            predictions: dict[str, np.ndarray] = {}
            truth_physical: np.ndarray | None = None
            for model_name, model, checkpoint_path in model_specs:
                result, prediction_physical, truth_physical = evaluate_model(
                    model,
                    input_tensor,
                    truth_tensor,
                    resized_valid_mask,
                    min_value,
                    max_value,
                )
                predictions[model_name] = prediction_physical
                level_mae, level_rmse, level_bias = depth_metrics_by_level(prediction_physical, truth_physical)
                depth_profile_accumulator[input_kind][model_name]["mae"].append(level_mae)
                depth_profile_accumulator[input_kind][model_name]["rmse"].append(level_rmse)
                depth_profile_accumulator[input_kind][model_name]["bias"].append(level_bias)
                for depth_index, (mae_value, rmse_value, bias_value) in enumerate(
                    zip(level_mae, level_rmse, level_bias)
                ):
                    depth_profile_writer.writerow(
                        {
                            "input_date": input_date,
                            "truth_date": truth_date,
                            "lead_days": args.lead_days,
                            "input_kind": input_kind,
                            "normalization_reference": args.normalization_reference,
                            "model": model_name,
                            "checkpoint": str(checkpoint_path),
                            "input_file": str(input_path),
                            "truth_file": str(reanalysis_by_date[truth_date]),
                            "depth_index": depth_index,
                            "mae": float(mae_value),
                            "rmse": float(rmse_value),
                            "bias_pred_minus_truth": float(bias_value),
                        }
                    )
                depth_profile_file.flush()
                row = {
                    "input_date": input_date,
                    "truth_date": truth_date,
                    "lead_days": args.lead_days,
                    "input_kind": input_kind,
                    "normalization_reference": args.normalization_reference,
                    "model": model_name,
                    "checkpoint": str(checkpoint_path),
                    "input_file": str(input_path),
                    "truth_file": str(reanalysis_by_date[truth_date]),
                    "shape": str(tuple(truth_tensor.shape[1:])),
                    "input_min": min_value,
                    "input_max": max_value,
                    **result,
                }
                rows.append(row)
                date_results.append(row)
            date_results_by_input[input_kind] = date_results
            if truth_physical is not None:
                fig3_payload[input_kind] = {
                    "initial": input_physical,
                    "truth": truth_physical,
                    "predictions": dict(predictions),
                }

            before_row, after_row = date_results
            delta_rmse = float(after_row["rmse"]) - float(before_row["rmse"])
            delta_mae = float(after_row["mae"]) - float(before_row["mae"])
            print(
                f"{input_kind} {input_date} -> {truth_date}: "
                f"{args.before_name} RMSE={before_row['rmse']:.4f}, MAE={before_row['mae']:.4f}; "
                f"{args.after_name} RMSE={after_row['rmse']:.4f}, MAE={after_row['mae']:.4f}; "
                f"after-before delta_RMSE={delta_rmse:+.4f}, delta_MAE={delta_mae:+.4f}"
            )

            if args.plot and truth_physical is not None:
                should_plot_case = args.plot_limit < 0 or date_index < args.plot_limit
                depth_path = (
                    plot_dir
                    / input_kind
                    / "depth_profiles"
                    / f"{input_date}_to_{truth_date}_depth_profile.png"
                )
                if should_plot_case:
                    plot_depth_profile(
                        depth_path,
                        input_date,
                        truth_date,
                        args.before_name,
                        args.after_name,
                        truth_physical,
                        predictions[args.before_name],
                        predictions[args.after_name],
                    )
                    saved_figures.append(depth_path)

                if should_plot_case:
                    spatial_path = (
                        plot_dir
                        / input_kind
                        / "spatial"
                        / f"{input_date}_to_{truth_date}_depth{args.plot_depth_index:02d}.png"
                    )
                    plot_spatial_comparison(
                        spatial_path,
                        input_date,
                        truth_date,
                        args.plot_depth_index,
                        args.before_name,
                        args.after_name,
                        truth_physical,
                        predictions[args.before_name],
                        predictions[args.after_name],
                    )
                    saved_figures.append(spatial_path)

        if (
            args.fig3_case_output is not None
            and "reanalysis_init" in fig3_payload
            and "analysis_init" in fig3_payload
        ):
            plot_fig3_case_layout(
                args.fig3_case_output,
                input_date,
                truth_date,
                args.plot_depth_index,
                args.before_name,
                args.after_name,
                fig3_payload["reanalysis_init"]["initial"],
                fig3_payload["analysis_init"]["initial"],
                fig3_payload["reanalysis_init"]["truth"],
                fig3_payload["analysis_init"]["truth"],
                fig3_payload["reanalysis_init"]["predictions"],
                fig3_payload["analysis_init"]["predictions"],
                args.lon_min,
                args.lon_max,
                args.lat_min,
                args.lat_max,
            )
            saved_figures.append(args.fig3_case_output)
            if args.fig3_case_output.suffix.lower() != ".png":
                saved_figures.append(args.fig3_case_output.with_suffix(".png"))
            print(f"Saved Fig. 3 case layout: {args.fig3_case_output}")

        if "reanalysis_init" in date_results_by_input and "analysis_init" in date_results_by_input:
            reanalysis_rows = {row["model"]: row for row in date_results_by_input["reanalysis_init"]}
            analysis_rows = {row["model"]: row for row in date_results_by_input["analysis_init"]}
            for model_name in (args.before_name, args.after_name):
                if model_name not in reanalysis_rows or model_name not in analysis_rows:
                    continue
                real_delta_rmse = float(analysis_rows[model_name]["rmse"]) - float(reanalysis_rows[model_name]["rmse"])
                real_delta_mae = float(analysis_rows[model_name]["mae"]) - float(reanalysis_rows[model_name]["mae"])
                print(
                    f"  real-init degradation {model_name}: "
                    f"analysis-minus-reanalysis delta_RMSE={real_delta_rmse:+.4f}, "
                    f"delta_MAE={real_delta_mae:+.4f}"
                )

    depth_profile_file.close()
    print(f"Saved depth-profile metrics CSV: {depth_profile_csv}")

    output_csv = args.output_csv or (
        args.output_dir / f"model_compare_analysis_init_reanalysis_truth_{args.variable}_lead{args.lead_days}.csv"
    )
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print()
    print("Overall summary")
    for input_kind in input_kind_names:
        input_rows = [row for row in rows if row["input_kind"] == input_kind]
        if not input_rows:
            continue
        print(f"{input_kind}:")
        summarize(input_rows, args.before_name)
        summarize(input_rows, args.after_name)

        before_rmse = np.asarray([float(row["rmse"]) for row in input_rows if row["model"] == args.before_name])
        after_rmse = np.asarray([float(row["rmse"]) for row in input_rows if row["model"] == args.after_name])
        before_mae = np.asarray([float(row["mae"]) for row in input_rows if row["model"] == args.before_name])
        after_mae = np.asarray([float(row["mae"]) for row in input_rows if row["model"] == args.after_name])
        print(
            f"  After-minus-before: mean_delta_RMSE={np.nanmean(after_rmse - before_rmse):+.4f}, "
            f"mean_delta_MAE={np.nanmean(after_mae - before_mae):+.4f}"
        )
    print(f"Saved CSV: {output_csv}")

    degradation_rows = build_degradation_rows(rows)
    if degradation_rows:
        degradation_csv = args.degradation_csv or (
            args.output_dir / f"model_compare_real_init_error_degradation_{args.variable}_lead{args.lead_days}.csv"
        )
        degradation_csv.parent.mkdir(parents=True, exist_ok=True)
        with degradation_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(degradation_rows[0].keys()))
            writer.writeheader()
            writer.writerows(degradation_rows)
        print_degradation_summary(degradation_rows, args.before_name, args.after_name)
        print(f"Saved degradation CSV: {degradation_csv}")

    if args.plot:
        depth_index = np.arange(depth_levels)
        for input_kind in input_kind_names:
            if not depth_profile_accumulator[input_kind][args.before_name]["rmse"]:
                continue
            mean_profile_path = plot_dir / input_kind / "mean_depth_profile.png"
            fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2), constrained_layout=True)
            for model_name in (args.before_name, args.after_name):
                mean_rmse = np.nanmean(np.stack(depth_profile_accumulator[input_kind][model_name]["rmse"]), axis=0)
                mean_mae = np.nanmean(np.stack(depth_profile_accumulator[input_kind][model_name]["mae"]), axis=0)
                mean_bias = np.nanmean(np.stack(depth_profile_accumulator[input_kind][model_name]["bias"]), axis=0)
                axes[0].plot(depth_index, mean_rmse, label=model_name)
                axes[1].plot(depth_index, mean_mae, label=model_name)
                axes[2].plot(depth_index, mean_bias, label=model_name)

            axes[0].set_title("Mean RMSE by depth")
            axes[0].set_xlabel("Depth index")
            axes[0].set_ylabel("RMSE")
            axes[1].set_title("Mean MAE by depth")
            axes[1].set_xlabel("Depth index")
            axes[1].set_ylabel("MAE")
            axes[2].set_title("Mean bias by depth")
            axes[2].set_xlabel("Depth index")
            axes[2].set_ylabel("Prediction - truth")
            for ax in axes:
                ax.legend()
            fig.suptitle(f"{input_kind}: mean over {len(candidate_dates)} dates")
            mean_profile_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(mean_profile_path, dpi=170)
            plt.close(fig)
            saved_figures.append(mean_profile_path)
        print(f"Saved figures: {plot_dir}")


if __name__ == "__main__":
    main()
