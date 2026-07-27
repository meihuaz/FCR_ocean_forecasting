from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr


ANALYSIS_ROOT = Path(r"D:\Data\glory_analysis")
REANALYSIS_ROOT = Path(r"D:\Data\glory_reanalysis")
OUTPUT_DIR = Path(r"E:\Data\glory_compare_analysis_reanalysis")

DATE_DASH_RE = re.compile(r"(19|20)\d{2}-\d{2}-\d{2}")
DATE_COMPACT_RE = re.compile(r"(19|20)\d{6}")
WINDOWS_DRIVE_RE = re.compile(r"^([A-Za-z]):[\\/](.*)$")


def normalize_path(path: Path) -> Path:
    """Translate Windows drive paths to WSL mount paths when needed."""
    text = str(path)
    match = WINDOWS_DRIVE_RE.match(text)
    if match is None or path.exists():
        return path

    drive = match.group(1).lower()
    rest = match.group(2).replace("\\", "/")
    candidate = Path("/mnt") / drive / rest
    if candidate.exists() or candidate.parent.exists() or (Path("/mnt") / drive).exists():
        return candidate
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare same-date GLORY analysis and reanalysis NetCDF files. "
            "Dates are matched from filenames."
        )
    )
    parser.add_argument("--analysis_root", default=ANALYSIS_ROOT, type=Path)
    parser.add_argument("--reanalysis_root", default=REANALYSIS_ROOT, type=Path)
    parser.add_argument("--output_dir", default=OUTPUT_DIR, type=Path)
    parser.add_argument("--variable", default="thetao")
    parser.add_argument("--date_start", default=None, help="Inclusive start date, e.g. 2022-06-01.")
    parser.add_argument("--date_end", default=None, help="Inclusive end date, e.g. 2022-06-30.")
    parser.add_argument("--limit", default=None, type=int, help="Only compare the first N matched dates.")
    parser.add_argument(
        "--plot",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save one comparison figure for each matched date.",
    )
    parser.add_argument(
        "--plot_limit",
        default=None,
        type=int,
        help="Only save figures for the first N matched dates. Defaults to all compared dates.",
    )
    parser.add_argument(
        "--plot_dir",
        default=None,
        type=Path,
        help="Directory for comparison figures. Defaults to <output_dir>/figures.",
    )
    parser.add_argument(
        "--diff_abs_max",
        default=None,
        type=float,
        help="Symmetric color limit for analysis minus reanalysis. Defaults to the date's 99th percentile absolute difference.",
    )

    parser.add_argument("--lon_min", default=100.0, type=float)
    parser.add_argument("--lon_max", default=180.0, type=float)
    parser.add_argument("--lat_min", default=0.0, type=float)
    parser.add_argument("--lat_max", default=60.0, type=float)
    parser.add_argument(
        "--full_domain",
        action="store_true",
        help="Compare the full longitude/latitude domain instead of the default northwest Pacific region.",
    )

    parser.add_argument("--depth", default=0.49402499198913574, type=float)
    parser.add_argument("--all_depths", action="store_true")
    parser.add_argument(
        "--small_rmse",
        default=0.1,
        type=float,
        help="RMSE threshold used for the simple final judgement.",
    )
    parser.add_argument(
        "--moderate_rmse",
        default=0.5,
        type=float,
        help="RMSE threshold used for the simple final judgement.",
    )
    args = parser.parse_args()
    args.analysis_root = normalize_path(args.analysis_root)
    args.reanalysis_root = normalize_path(args.reanalysis_root)
    args.output_dir = normalize_path(args.output_dir)
    if args.plot_dir is not None:
        args.plot_dir = normalize_path(args.plot_dir)
    return args


def date_from_filename(path: Path) -> str | None:
    dash_match = DATE_DASH_RE.search(path.name)
    if dash_match:
        return dash_match.group(0)

    compact_match = DATE_COMPACT_RE.search(path.name)
    if compact_match:
        text = compact_match.group(0)
        return f"{text[:4]}-{text[4:6]}-{text[6:8]}"

    return None


def build_date_index(root: Path) -> dict[str, Path]:
    if not root.exists():
        raise FileNotFoundError(f"Path does not exist: {root}")

    files = sorted(root.rglob("*.nc"))
    if not files:
        raise FileNotFoundError(f"No .nc files found under {root}")

    by_date: dict[str, Path] = {}
    duplicates: dict[str, list[Path]] = {}
    for path in files:
        date_text = date_from_filename(path)
        if date_text is None:
            continue
        if date_text in by_date:
            duplicates.setdefault(date_text, [by_date[date_text]]).append(path)
            continue
        by_date[date_text] = path

    if duplicates:
        print(f"Warning: found duplicate dates under {root}; using the first sorted file for each date.")
        for date_text, paths in sorted(duplicates.items())[:5]:
            print(f"  {date_text}: {len(paths)} files")

    if not by_date:
        raise FileNotFoundError(f"No dated .nc files found under {root}")
    return by_date


def filter_dates(dates: list[str], date_start: str | None, date_end: str | None, limit: int | None) -> list[str]:
    selected = [
        date_text
        for date_text in sorted(dates)
        if (date_start is None or date_text >= date_start)
        and (date_end is None or date_text <= date_end)
    ]
    if limit is not None:
        selected = selected[:limit]
    if not selected:
        raise FileNotFoundError("No same-date file pairs matched the requested date range.")
    return selected


def open_dataset(path: Path) -> xr.Dataset:
    last_error: Exception | None = None
    for engine in ("h5netcdf", "scipy", None):
        try:
            return xr.open_dataset(path, engine=engine, mask_and_scale=True)
        except Exception as exc:
            last_error = exc
    raise OSError(f"Cannot open NetCDF file: {path}") from last_error


def coord_name(data: xr.DataArray, candidates: tuple[str, ...]) -> str | None:
    for name in candidates:
        if name in data.coords or name in data.dims:
            return name
    return None


def select_range(data: xr.DataArray, name: str, min_value: float, max_value: float) -> xr.DataArray:
    coord = data[name]
    first = float(coord.values[0])
    last = float(coord.values[-1])
    if first <= last:
        return data.sel({name: slice(min_value, max_value)})
    return data.sel({name: slice(max_value, min_value)})


def select_subset(ds: xr.Dataset, variable: str, args: argparse.Namespace) -> xr.DataArray:
    if variable not in ds:
        raise KeyError(f"{variable!r} not found. Available variables: {list(ds.data_vars)}")

    data = ds[variable]

    time_name = coord_name(data, ("time",))
    if time_name is not None and time_name in data.dims and data.sizes[time_name] == 1:
        data = data.isel({time_name: 0})

    depth_name = coord_name(data, ("depth", "elevation"))
    if depth_name is not None and depth_name in data.dims and not args.all_depths:
        data = data.sel({depth_name: args.depth}, method="nearest")

    if not args.full_domain:
        lat_name = coord_name(data, ("latitude", "lat"))
        lon_name = coord_name(data, ("longitude", "lon"))
        if lat_name is not None:
            data = select_range(data, lat_name, args.lat_min, args.lat_max)
        if lon_name is not None:
            data = select_range(data, lon_name, args.lon_min, args.lon_max)

    return data


def same_shape_values(analysis: xr.DataArray, reanalysis: xr.DataArray) -> tuple[np.ndarray, np.ndarray]:
    a = np.asarray(analysis.values, dtype=np.float64)
    r = np.asarray(reanalysis.values, dtype=np.float64)
    if a.shape == r.shape:
        return a, r

    common_shape = tuple(min(x, y) for x, y in zip(a.shape, r.shape))
    if len(a.shape) != len(r.shape) or any(size == 0 for size in common_shape):
        raise ValueError(f"Shape mismatch: analysis {a.shape}, reanalysis {r.shape}")

    slices = tuple(slice(0, size) for size in common_shape)
    print(f"Warning: shape mismatch analysis {a.shape}, reanalysis {r.shape}; trimming to {common_shape}.")
    return a[slices], r[slices]


def compare_values(analysis: np.ndarray, reanalysis: np.ndarray) -> dict[str, float | int]:
    mask = np.isfinite(analysis) & np.isfinite(reanalysis)
    valid_count = int(mask.sum())
    total_count = int(mask.size)
    if valid_count == 0:
        return {
            "total_count": total_count,
            "valid_count": 0,
            "valid_fraction": 0.0,
            "analysis_mean": np.nan,
            "reanalysis_mean": np.nan,
            "bias_analysis_minus_reanalysis": np.nan,
            "mae": np.nan,
            "rmse": np.nan,
            "median_abs_diff": np.nan,
            "p95_abs_diff": np.nan,
            "max_abs_diff": np.nan,
            "corr": np.nan,
        }

    a = analysis[mask]
    r = reanalysis[mask]
    diff = a - r
    abs_diff = np.abs(diff)

    if valid_count > 1 and np.nanstd(a) > 0 and np.nanstd(r) > 0:
        corr = float(np.corrcoef(a, r)[0, 1])
    else:
        corr = np.nan

    return {
        "total_count": total_count,
        "valid_count": valid_count,
        "valid_fraction": valid_count / total_count,
        "analysis_mean": float(np.nanmean(a)),
        "reanalysis_mean": float(np.nanmean(r)),
        "bias_analysis_minus_reanalysis": float(np.nanmean(diff)),
        "mae": float(np.nanmean(abs_diff)),
        "rmse": float(np.sqrt(np.nanmean(diff * diff))),
        "median_abs_diff": float(np.nanmedian(abs_diff)),
        "p95_abs_diff": float(np.nanpercentile(abs_diff, 95)),
        "max_abs_diff": float(np.nanmax(abs_diff)),
        "corr": corr,
    }


def compare_by_depth(
    date_text: str,
    analysis_data: xr.DataArray,
    analysis_values: np.ndarray,
    reanalysis_values: np.ndarray,
) -> list[dict[str, object]]:
    if "depth" not in analysis_data.dims:
        return []

    depth_axis = analysis_data.dims.index("depth")
    depth_values = np.asarray(analysis_data["depth"].values, dtype=np.float64)
    rows: list[dict[str, object]] = []

    for depth_index, depth_value in enumerate(depth_values):
        selector = [slice(None)] * analysis_values.ndim
        selector[depth_axis] = depth_index
        selector_tuple = tuple(selector)
        stats = compare_values(analysis_values[selector_tuple], reanalysis_values[selector_tuple])
        rows.append(
            {
                "date": date_text,
                "depth_index": depth_index,
                "depth_m": float(depth_value),
                **stats,
            }
        )
    return rows


def get_lon_lat(data: xr.DataArray, shape: tuple[int, ...]) -> tuple[np.ndarray | None, np.ndarray | None]:
    if len(shape) < 2:
        return None, None

    lat_name = coord_name(data, ("latitude", "lat"))
    lon_name = coord_name(data, ("longitude", "lon"))
    if lat_name is None or lon_name is None:
        return None, None

    lat = np.asarray(data[lat_name].values, dtype=np.float64)
    lon = np.asarray(data[lon_name].values, dtype=np.float64)
    if lat.ndim != 1 or lon.ndim != 1:
        return None, None
    if lat.size < shape[-2] or lon.size < shape[-1]:
        return None, None
    return lon[: shape[-1]], lat[: shape[-2]]


def image_extent(lon: np.ndarray | None, lat: np.ndarray | None) -> tuple[float, float, float, float] | None:
    if lon is None or lat is None or lon.size < 2 or lat.size < 2:
        return None

    lon_step = float(np.nanmedian(np.diff(lon)))
    lat_step = float(np.nanmedian(np.diff(lat)))
    return (
        float(lon[0] - lon_step / 2),
        float(lon[-1] + lon_step / 2),
        float(lat[0] - lat_step / 2),
        float(lat[-1] + lat_step / 2),
    )


def color_limits(values: np.ndarray, percentile: float = 1.0) -> tuple[float, float]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 0.0, 1.0
    vmin, vmax = np.nanpercentile(finite, [percentile, 100.0 - percentile])
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
        vmin = float(np.nanmin(finite))
        vmax = float(np.nanmax(finite))
    if vmin == vmax:
        margin = max(abs(vmin) * 0.01, 1.0)
        return float(vmin - margin), float(vmax + margin)
    return float(vmin), float(vmax)


def plot_comparison(
    output_path: Path,
    date_text: str,
    variable: str,
    analysis_data: xr.DataArray,
    analysis_values: np.ndarray,
    reanalysis_values: np.ndarray,
    stats: dict[str, float | int],
    diff_abs_max: float | None,
) -> None:
    if analysis_values.ndim != 2:
        print(f"Skipping figure for {date_text}: expected 2D data after selection, got {analysis_values.shape}.")
        return

    diff = analysis_values - reanalysis_values
    combined = np.concatenate(
        [
            analysis_values[np.isfinite(analysis_values)].reshape(-1),
            reanalysis_values[np.isfinite(reanalysis_values)].reshape(-1),
        ]
    )
    vmin, vmax = color_limits(combined)
    if diff_abs_max is None:
        finite_abs = np.abs(diff[np.isfinite(diff)])
        diff_limit = float(np.nanpercentile(finite_abs, 99)) if finite_abs.size else 1.0
    else:
        diff_limit = diff_abs_max
    if not np.isfinite(diff_limit) or diff_limit <= 0:
        diff_limit = 1.0

    lon, lat = get_lon_lat(analysis_data, analysis_values.shape)
    extent = image_extent(lon, lat)
    origin = "lower" if lat is not None and lat[0] <= lat[-1] else "upper"

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), constrained_layout=True)
    panels = [
        ("Analysis", analysis_values, "viridis", vmin, vmax),
        ("Reanalysis", reanalysis_values, "viridis", vmin, vmax),
        ("Analysis - Reanalysis", diff, "RdBu_r", -diff_limit, diff_limit),
    ]
    for ax, (title, values, cmap, panel_vmin, panel_vmax) in zip(axes, panels):
        image = ax.imshow(
            values,
            origin=origin,
            extent=extent,
            cmap=cmap,
            vmin=panel_vmin,
            vmax=panel_vmax,
            aspect="auto",
        )
        ax.set_title(title)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(
        f"{date_text} {variable} | RMSE={stats['rmse']:.4f}, "
        f"MAE={stats['mae']:.4f}, bias={stats['bias_analysis_minus_reanalysis']:.4f}, "
        f"corr={stats['corr']:.5f}"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def judgement(mean_rmse: float, small_rmse: float, moderate_rmse: float) -> str:
    if not np.isfinite(mean_rmse):
        return "无法判断：没有有效重叠数据。"
    if mean_rmse <= small_rmse:
        return f"差别较小：平均 RMSE <= {small_rmse:g}。"
    if mean_rmse <= moderate_rmse:
        return f"差别中等：平均 RMSE 在 {small_rmse:g} 到 {moderate_rmse:g} 之间。"
    return f"差别较大：平均 RMSE > {moderate_rmse:g}。"


def main() -> None:
    args = parse_args()
    analysis_by_date = build_date_index(args.analysis_root)
    reanalysis_by_date = build_date_index(args.reanalysis_root)
    common_dates = filter_dates(
        list(set(analysis_by_date) & set(reanalysis_by_date)),
        args.date_start,
        args.date_end,
        args.limit,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    domain_tag = "full_domain" if args.full_domain else "nw_pacific"
    depth_tag = "all_depths" if args.all_depths else "surface"
    output_csv = args.output_dir / f"compare_{args.variable}_{domain_tag}_{depth_tag}.csv"
    output_by_depth_csv = args.output_dir / f"compare_{args.variable}_{domain_tag}_by_depth.csv"
    plot_dir = args.plot_dir or (args.output_dir / "figures" / f"{args.variable}_{domain_tag}_{depth_tag}")

    print(f"Analysis files with dates: {len(analysis_by_date)}")
    print(f"Reanalysis files with dates: {len(reanalysis_by_date)}")
    print(f"Matched same-date pairs: {len(common_dates)}")
    print(f"Variable: {args.variable}")
    if args.full_domain:
        print("Domain: full grid")
    else:
        print(f"Domain: lon {args.lon_min} to {args.lon_max}, lat {args.lat_min} to {args.lat_max}")
    print("Depth: all levels" if args.all_depths else f"Depth: nearest to {args.depth} m")
    print()

    rows: list[dict[str, object]] = []
    depth_rows: list[dict[str, object]] = []
    saved_figures: list[Path] = []
    for date_index, date_text in enumerate(common_dates):
        analysis_path = analysis_by_date[date_text]
        reanalysis_path = reanalysis_by_date[date_text]
        with open_dataset(analysis_path) as analysis_ds, open_dataset(reanalysis_path) as reanalysis_ds:
            analysis_data = select_subset(analysis_ds, args.variable, args)
            reanalysis_data = select_subset(reanalysis_ds, args.variable, args)
            analysis_values, reanalysis_values = same_shape_values(analysis_data, reanalysis_data)
            stats = compare_values(analysis_values, reanalysis_values)
            if args.all_depths:
                depth_rows.extend(compare_by_depth(date_text, analysis_data, analysis_values, reanalysis_values))
            if args.plot and (args.plot_limit is None or date_index < args.plot_limit):
                figure_path = plot_dir / f"{date_text}_{args.variable}_{domain_tag}_{depth_tag}.png"
                plot_comparison(
                    figure_path,
                    date_text,
                    args.variable,
                    analysis_data,
                    analysis_values,
                    reanalysis_values,
                    stats,
                    args.diff_abs_max,
                )
                if figure_path.exists():
                    saved_figures.append(figure_path)

        row: dict[str, object] = {
            "date": date_text,
            "analysis_file": str(analysis_path),
            "reanalysis_file": str(reanalysis_path),
            "shape": str(analysis_values.shape),
            **stats,
        }
        rows.append(row)

        print(
            f"{date_text}: shape={analysis_values.shape}, "
            f"bias={stats['bias_analysis_minus_reanalysis']:.4f}, "
            f"MAE={stats['mae']:.4f}, RMSE={stats['rmse']:.4f}, "
            f"p95_abs={stats['p95_abs_diff']:.4f}, corr={stats['corr']:.5f}"
        )

    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    if depth_rows:
        with output_by_depth_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(depth_rows[0].keys()))
            writer.writeheader()
            writer.writerows(depth_rows)

    rmse_values = np.array([float(row["rmse"]) for row in rows], dtype=np.float64)
    mae_values = np.array([float(row["mae"]) for row in rows], dtype=np.float64)
    bias_values = np.array([float(row["bias_analysis_minus_reanalysis"]) for row in rows], dtype=np.float64)
    corr_values = np.array([float(row["corr"]) for row in rows], dtype=np.float64)

    print()
    print("Overall summary")
    print(f"Dates compared: {len(rows)}")
    print(f"Mean bias analysis-reanalysis: {np.nanmean(bias_values):.4f}")
    print(f"Mean MAE: {np.nanmean(mae_values):.4f}")
    print(f"Mean RMSE: {np.nanmean(rmse_values):.4f}")
    print(f"Mean corr: {np.nanmean(corr_values):.5f}")
    print(judgement(float(np.nanmean(rmse_values)), args.small_rmse, args.moderate_rmse))
    print(f"Saved CSV: {output_csv}")
    if saved_figures:
        print(f"Saved figures: {plot_dir}")
    if depth_rows:
        print(f"Saved by-depth CSV: {output_by_depth_csv}")

        by_depth: dict[int, dict[str, list[float]]] = {}
        depth_lookup: dict[int, float] = {}
        for row in depth_rows:
            depth_index = int(row["depth_index"])
            by_depth.setdefault(depth_index, {"rmse": [], "mae": [], "bias": []})
            by_depth[depth_index]["rmse"].append(float(row["rmse"]))
            by_depth[depth_index]["mae"].append(float(row["mae"]))
            by_depth[depth_index]["bias"].append(float(row["bias_analysis_minus_reanalysis"]))
            depth_lookup[depth_index] = float(row["depth_m"])

        depth_summary = [
            {
                "depth_index": depth_index,
                "depth_m": depth_lookup[depth_index],
                "mean_rmse": float(np.nanmean(values["rmse"])),
                "mean_mae": float(np.nanmean(values["mae"])),
                "mean_bias": float(np.nanmean(values["bias"])),
            }
            for depth_index, values in by_depth.items()
            if np.isfinite(values["rmse"]).any()
        ]
        depth_summary.sort(key=lambda row: row["mean_rmse"], reverse=True)
        print("Top 5 depths by mean RMSE:")
        for row in depth_summary[:5]:
            print(
                f"  depth_index={row['depth_index']:02d}, depth={row['depth_m']:.2f} m, "
                f"RMSE={row['mean_rmse']:.4f}, MAE={row['mean_mae']:.4f}, "
                f"bias={row['mean_bias']:.4f}"
            )


if __name__ == "__main__":
    main()
