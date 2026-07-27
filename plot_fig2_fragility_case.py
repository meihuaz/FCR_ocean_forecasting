import argparse
import os
from datetime import datetime, timedelta

import glob
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import colors
from matplotlib import rcParams
import numpy as np
import torch
import torch.nn.functional as F
import yaml

from adversarial_generation import PGDAttack
from model.model_factory import build_model, extract_prediction
from utils.metric import MAELoss

DOMAIN_LON = (100.0, 180.0)
DOMAIN_LAT = (0.0, 60.0)
DEG_C = "\N{DEGREE SIGN}C"
REPO_ROOT = Path(__file__).resolve().parent

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


class EN4GLORYDatasetTestWithMask:

    def __init__(self, glory_path, split="test", mean_prefix=None):
        self.glory_files = self._get_files(glory_path)
        if mean_prefix:
            self.dir = mean_prefix
        else:
            self.dir = os.path.join(glory_path, "mercatorglorys12v1_gl12_mean_")

    def _get_files(self, path):
        files = []
        for file in os.listdir(path):
            file_path = os.path.join(path, file)
            if os.path.isfile(file_path):
                files.append(file_path)
        return sorted(files)

    def __len__(self):
        return len(self.glory_files)

    def _resolve_pair(self, idx):
        glory_path = self.glory_files[idx]
        date_str = glory_path.split("/")[-1].split("_")[-2]
        date = datetime.strptime(date_str, "%Y%m%d") - timedelta(days=7)
        formatted_date = date.strftime("%Y%m%d")
        path = self.dir + formatted_date
        glory_filename = glob.glob(f"{path}*")

        while len(glory_filename) != 1:
            idx = (idx + 1) % len(self.glory_files)
            glory_path = self.glory_files[idx]
            date_str = glory_path.split("/")[-1].split("_")[-2]
            date = datetime.strptime(date_str, "%Y%m%d") - timedelta(days=7)
            formatted_date = date.strftime("%Y%m%d")
            path = self.dir + formatted_date
            glory_filename = glob.glob(f"{path}*")

        return glory_filename[0], glory_path, idx

    def get_scale(self, idx):
        glory_filename, _, _ = self._resolve_pair(idx)
        glory_data = np.load(glory_filename)
        min_val = np.nanmin(glory_data)
        max_val = np.nanmax(glory_data)
        return float(min_val), float(max_val)

    def __getitem__(self, idx):
        glory_filename, glory_path, _ = self._resolve_pair(idx)
        glory_data = np.load(glory_filename)
        min_val = np.nanmin(glory_data)
        max_val = np.nanmax(glory_data)
        normalized_glory_data = (glory_data - min_val) / (max_val - min_val)
        normalized_glory_data[np.isnan(normalized_glory_data)] = 0

        gt = np.load(glory_path)
        valid_mask = ~np.isnan(gt)
        normalized_gt = (gt - min_val) / (max_val - min_val)
        normalized_gt[np.isnan(normalized_gt)] = 0
        return normalized_glory_data, normalized_gt, valid_mask, float(min_val), float(max_val)


def preprocess_sample(glory_batch, gt, valid_mask, img_size, device):
    glory_batch = torch.from_numpy(glory_batch).unsqueeze(0)
    gt = torch.from_numpy(gt).unsqueeze(0)
    valid_mask = torch.from_numpy(valid_mask).unsqueeze(0).float()

    bg_field = glory_batch[:, :, :, :img_size[2]]
    bg_field[torch.isnan(bg_field)] = 0
    bg_field = bg_field.float().to(device).permute(0, 3, 1, 2)
    bg_field = F.interpolate(bg_field,
                             size=(img_size[0], img_size[1]),
                             mode="nearest")

    gt = gt[:, :, :, :img_size[2]]
    valid_mask = valid_mask[:, :, :, :img_size[2]]
    gt[torch.isnan(gt)] = 0
    gt = gt.float().to(device).permute(0, 3, 1, 2)
    valid_mask = valid_mask.float().to(device).permute(0, 3, 1, 2)
    gt = F.interpolate(gt, size=(img_size[0], img_size[1]),
                       mode="nearest").permute(0, 2, 3, 1)
    valid_mask = F.interpolate(
        valid_mask,
        size=(img_size[0], img_size[1]),
        mode="nearest",
    ).permute(0, 2, 3, 1)
    return bg_field, gt, valid_mask


def load_model_from_checkpoint(yaml_config, checkpoint_path, device):
    with open(yaml_config) as f:
        params = yaml.safe_load(f)

    model = build_model(params)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state = checkpoint["model_state"] if isinstance(
        checkpoint, dict) and "model_state" in checkpoint else checkpoint
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model, params


def rmse(pred, target, mask):
    diff = ((pred - target)**2) * mask
    mse = diff.sum() / mask.sum().clamp_min(1.0)
    return torch.sqrt(mse).item()


def masked_numpy(field, mask):
    out = np.array(field, copy=True)
    out[mask == 0] = np.nan
    return out


def index_to_lon_lat(x_idx, y_idx, width, height):
    lon = DOMAIN_LON[0] + (x_idx / width) * (DOMAIN_LON[1] - DOMAIN_LON[0])
    lat = DOMAIN_LAT[0] + (y_idx / height) * (DOMAIN_LAT[1] - DOMAIN_LAT[0])
    return lon, lat


def format_lon_tick(value):
    value = int(value)
    if value == 0:
        return "0\N{DEGREE SIGN}"
    hemi = "E" if value > 0 else "W"
    return f"{abs(value)}\N{DEGREE SIGN}{hemi}"


def format_lat_tick(value):
    value = int(value)
    if value == 0:
        return "0\N{DEGREE SIGN}"
    hemi = "N" if value > 0 else "S"
    return f"{abs(value)}\N{DEGREE SIGN}{hemi}"


def compute_roi(improvement_map, std_abs_error, mask, box_size):
    valid = mask > 0
    candidate = np.array(improvement_map, copy=True)
    candidate[~valid] = -np.inf
    if np.isfinite(candidate).any():
        y_idx, x_idx = np.unravel_index(np.argmax(candidate), candidate.shape)
    else:
        fallback = np.array(std_abs_error, copy=True)
        fallback[~valid] = -np.inf
        y_idx, x_idx = np.unravel_index(np.argmax(fallback), fallback.shape)
    return x_idx, y_idx, box_size


def add_roi_rectangle(ax, roi, shape):
    x_idx, y_idx, box_size = roi
    height, width = shape

    x0 = max(0, x_idx - box_size // 2)
    y0 = max(0, y_idx - box_size // 2)
    x1 = min(width, x0 + box_size)
    y1 = min(height, y0 + box_size)

    lon0, lat0 = index_to_lon_lat(x0, y0, width, height)
    lon1, lat1 = index_to_lon_lat(x1, y1, width, height)
    rect = patches.Rectangle(
        (lon0, lat0),
        lon1 - lon0,
        lat1 - lat0,
        linewidth=3.0,
        edgecolor="red",
        facecolor="none",
    )
    ax.add_patch(rect)


def load_or_generate_perturbation(
    sample_npz_path,
    regenerate,
    dataset,
    sample_index,
    standard_model,
    params,
    epsilon,
    alpha,
    steps,
    device,
):
    if sample_npz_path and os.path.exists(sample_npz_path) and not regenerate:
        data = np.load(sample_npz_path)
        cached_idx = int(data["idx"]) if "idx" in data.files else sample_index
        min_val, max_val = dataset.get_scale(cached_idx)
        if {"bg_field", "x_pert", "delta_star", "gt", "valid_mask"}.issubset(data.files):
            bg_field = torch.from_numpy(data["bg_field"]).float().to(device)
            x_pert = torch.from_numpy(data["x_pert"]).float().to(device)
            delta_star = torch.from_numpy(data["delta_star"]).float().to(device)
            gt = torch.from_numpy(data["gt"]).float().to(device)
            valid_mask = torch.from_numpy(data["valid_mask"]).float().to(device)
        elif {"x_clean", "x_adv", "delta", "y"}.issubset(data.files):
            bg_field = torch.from_numpy(data["x_clean"]).float().to(device)
            x_pert = torch.from_numpy(data["x_adv"]).float().to(device)
            delta_star = torch.from_numpy(data["delta"]).float().to(device)
            gt = torch.from_numpy(data["y"]).float().to(device)
            _, raw_gt, raw_valid_mask, _, _ = dataset[cached_idx]
            _, _, valid_mask = preprocess_sample(
                np.zeros_like(raw_gt),
                raw_gt,
                raw_valid_mask,
                params["img_size"],
                device,
            )
        else:
            raise KeyError(
                f"Unsupported sample npz format in {sample_npz_path}. "
                f"Available keys: {list(data.files)}"
            )
        return bg_field, x_pert, delta_star, gt, valid_mask, min_val, max_val, "loaded"

    glory_batch, gt, valid_mask, min_val, max_val = dataset[sample_index]
    bg_field, gt, valid_mask = preprocess_sample(
        glory_batch,
        gt,
        valid_mask,
        params["img_size"],
        device,
    )

    criterion = MAELoss()
    attacker = PGDAttack(
        standard_model,
        epsilon=epsilon,
        alpha=alpha,
        num_iter=steps,
        device=device,
    )
    x_pert = attacker.generate(bg_field, gt, criterion, random_start=False)
    delta_star = x_pert - bg_field
    return bg_field, x_pert, delta_star, gt, valid_mask, min_val, max_val, "generated"


def plot_case(
    bg_field,
    x_pert,
    delta_star,
    valid_mask,
    pred_std_clean,
    pred_std_pert,
    gt,
    rmse_std_clean,
    rmse_std_pert,
    min_val,
    max_val,
    sample_index,
    box_size,
    save_path,
):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    var_idx = 0
    mask_vis = valid_mask[0, :, :, var_idx].detach().cpu().numpy()
    dynamic_range = max_val - min_val

    def to_celsius(field):
        return field * dynamic_range + min_val

    def delta_to_celsius(delta):
        return delta * dynamic_range

    img_clean = masked_numpy(to_celsius(bg_field[0, var_idx].detach().cpu().numpy()),
                             mask_vis)
    img_pert = masked_numpy(to_celsius(x_pert[0, var_idx].detach().cpu().numpy()),
                            mask_vis)
    gt_vis = masked_numpy(to_celsius(gt[0, :, :, var_idx].detach().cpu().numpy()),
                          mask_vis)
    pred_std_clean_vis = masked_numpy(
        to_celsius(pred_std_clean[0, :, :, var_idx].detach().cpu().numpy()),
        mask_vis)
    pred_std_pert_vis = masked_numpy(
        to_celsius(pred_std_pert[0, :, :, var_idx].detach().cpu().numpy()),
        mask_vis)

    delta_vis = delta_to_celsius(delta_star[0, var_idx].detach().cpu().numpy())
    delta_vis[mask_vis == 0] = 0.0

    clean_abs_error = np.abs(pred_std_clean_vis - gt_vis)
    pert_abs_error = np.abs(pred_std_pert_vis - gt_vis)
    error_increase = pert_abs_error - clean_abs_error

    roi = compute_roi(error_increase, pert_abs_error, mask_vis, box_size)

    state_stack = np.stack([
        img_clean,
        img_pert,
        pred_std_clean_vis,
        pred_std_pert_vis,
    ])
    state_vmin = np.nanmin(state_stack)
    state_vmax = np.nanmax(state_stack)

    delta_lim = np.nanmax(np.abs(delta_vis))
    if not np.isfinite(delta_lim) or delta_lim == 0:
        delta_lim = 1e-6

    increase_lim = np.nanmax(np.abs(error_increase))
    if not np.isfinite(increase_lim) or increase_lim == 0:
        increase_lim = 1e-6

    cmap_state = plt.get_cmap("viridis").copy()
    cmap_state.set_bad("white")
    cmap_diff = plt.get_cmap("RdBu_r").copy()
    cmap_increase = plt.get_cmap("RdBu_r").copy()
    cmap_increase.set_bad("white")

    extent = (DOMAIN_LON[0], DOMAIN_LON[1], DOMAIN_LAT[0], DOMAIN_LAT[1])
    fig, axes = plt.subplots(2, 3, figsize=(20, 10), constrained_layout=True)

    state_kwargs = dict(cmap=cmap_state,
                        origin="lower",
                        extent=extent,
                        vmin=state_vmin,
                        vmax=state_vmax)
    diff_kwargs = dict(cmap=cmap_diff, origin="lower", extent=extent)

    im_a = axes[0, 0].imshow(img_clean, **state_kwargs)
    axes[0, 0].set_title("(a) Clean Initial Condition")

    im_b = axes[0, 1].imshow(img_pert, **state_kwargs)
    axes[0, 1].set_title("(b) Perturbed Initial Condition")

    im_c = axes[0, 2].imshow(delta_vis,
                             **diff_kwargs,
                             vmin=-0.024,
                             vmax=0.024)
    axes[0, 2].set_title("(c) Worst-case Perturbation")

    im_d = axes[1, 0].imshow(pred_std_clean_vis, **state_kwargs)
    axes[1,
         0].set_title(f"(d) Clean Forecast\nRMSE = {rmse_std_clean:.3f} {DEG_C}")
    add_roi_rectangle(axes[1, 0], roi, pred_std_pert_vis.shape)

    im_e = axes[1, 1].imshow(pred_std_pert_vis, **state_kwargs)
    axes[1,
         1].set_title(f"(e) Perturbed Forecast\nRMSE = {rmse_std_pert:.3f} {DEG_C}")
    add_roi_rectangle(axes[1, 1], roi, pred_std_pert_vis.shape)

    increase_norm = colors.TwoSlopeNorm(vmin=-increase_lim,
                                        vcenter=0.0,
                                        vmax=increase_lim)
    im_f = axes[1, 2].imshow(error_increase,
                             cmap=cmap_increase,
                             origin="lower",
                             extent=extent,
                             norm=increase_norm)
    axes[1,
         2].set_title("(f) Forecast Error Increase\n|Pert - GT| - |Clean - GT|")
    add_roi_rectangle(axes[1, 2], roi, error_increase.shape)

    for ax in axes.flat:
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        lon_ticks = [100, 120, 140, 160, 180]
        lat_ticks = [0, 15, 30, 45, 60]
        ax.set_xticks(lon_ticks)
        ax.set_yticks(lat_ticks)
        ax.set_xticklabels([format_lon_tick(t) for t in lon_ticks])
        ax.set_yticklabels([format_lat_tick(t) for t in lat_ticks])

    cb_a = fig.colorbar(im_a,
                        ax=axes[0, 0],
                        shrink=0.86,
                        pad=0.02,
                        label=f"Temperature ({DEG_C})")
    cb_b = fig.colorbar(im_b,
                        ax=axes[0, 1],
                        shrink=0.86,
                        pad=0.02,
                        label=f"Temperature ({DEG_C})")
    cb_c = fig.colorbar(im_c,
                        ax=axes[0, 2],
                        shrink=0.86,
                        pad=0.02,
                        label=f"Perturbation ({DEG_C})")
    cb_d = fig.colorbar(im_d,
                        ax=axes[1, 0],
                        shrink=0.86,
                        pad=0.02,
                        label=f"Temperature ({DEG_C})")
    cb_e = fig.colorbar(im_e,
                        ax=axes[1, 1],
                        shrink=0.86,
                        pad=0.02,
                        label=f"Temperature ({DEG_C})")
    cb_f = fig.colorbar(
        im_f,
        ax=axes[1, 2],
        shrink=0.86,
        pad=0.02,
        label=f"Error increase ({DEG_C})",
    )

    for cb in [cb_a, cb_b, cb_c, cb_d, cb_e, cb_f]:
        cb.ax.tick_params(labelsize=16)
        cb.set_label(cb.ax.get_ylabel(), fontsize=18)

    # fig.suptitle(
    #     f"Perturbation-induced Forecast Fragility Case Study (sample {sample_index})",
    #     fontsize=28,
    # )
    fig.savefig(save_path, bbox_inches="tight")
    root, ext = os.path.splitext(save_path)
    if ext.lower() != ".png":
        fig.savefig(f"{root}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Generate the updated 4x2 Fig. 2 case-study figure.")
    parser.add_argument(
        "--yaml_config",
        default=str(REPO_ROOT / "config" / "en4_Transformer_bg_model3.yaml"),
    )
    parser.add_argument(
        "--standard_checkpoint",
        default=str(REPO_ROOT / "ckpt" / "swinunet" / "best_ckpt.tar"),
    )
    parser.add_argument(
        "--sample_npz_path",
        default=str(
            REPO_ROOT
            / "figures"
            / "single_case_sensitivity"
            / "saved_samples"
            / "adv_sample_idx0_eps0.0005.npz"
        ),
    )
    parser.add_argument(
        "--glory_path",
        default="/mnt/f/zhishuai/Data/test_nw_pacific/",
        help="Directory containing target GLORY npy files for the selected split.",
    )
    parser.add_argument(
        "--glory_mean_prefix",
        default="",
        help=(
            "Prefix for the t-7 background files. Defaults to "
            "<glory_path>/mercatorglorys12v1_gl12_mean_."
        ),
    )
    parser.add_argument("--sample_index", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--epsilon", type=float, default=5e-4)
    parser.add_argument("--alpha", type=float, default=1e-4)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--box_size", type=int, default=40)
    parser.add_argument(
        "--save_path",
        default=str(
            REPO_ROOT
            / "sn-fcr_bundle_20260707"
            / "latex"
            / "figures"
            / "fig2_fragility_case"
            / "fig2_fragility_case_idx0_celsius1.pdf"
        ),
    )
    parser.add_argument(
        "--regenerate_perturbation",
        action="store_true",
        help=
        "Regenerate the perturbation with the standard model instead of reusing the saved sample.",
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    standard_model, params = load_model_from_checkpoint(
        args.yaml_config,
        args.standard_checkpoint,
        device,
    )
    dataset = EN4GLORYDatasetTestWithMask(
        args.glory_path,
        mean_prefix=args.glory_mean_prefix or None,
    )
    bg_field, x_pert, delta_star, gt, valid_mask, min_val, max_val, source = load_or_generate_perturbation(
        args.sample_npz_path,
        args.regenerate_perturbation,
        dataset,
        args.sample_index,
        standard_model,
        params,
        args.epsilon,
        args.alpha,
        args.steps,
        device,
    )

    with torch.no_grad():
        pred_std_clean = extract_prediction(standard_model(bg_field)).permute(
            0, 2, 3, 1)
        pred_std_pert = extract_prediction(standard_model(x_pert)).permute(
            0, 2, 3, 1)

    dynamic_range = max_val - min_val
    rmse_std_clean_norm = rmse(pred_std_clean, gt, valid_mask)
    rmse_std_pert_norm = rmse(pred_std_pert, gt, valid_mask)
    rmse_std_clean = rmse_std_clean_norm * dynamic_range
    rmse_std_pert = rmse_std_pert_norm * dynamic_range

    plot_case(
        bg_field=bg_field,
        x_pert=x_pert,
        delta_star=delta_star,
        valid_mask=valid_mask,
        pred_std_clean=pred_std_clean,
        pred_std_pert=pred_std_pert,
        gt=gt,
        rmse_std_clean=rmse_std_clean,
        rmse_std_pert=rmse_std_pert,
        min_val=min_val,
        max_val=max_val,
        sample_index=args.sample_index,
        box_size=args.box_size,
        save_path=args.save_path,
    )

    print(f"Perturbation source: {source}")
    print(f"Temperature range: min={min_val:.4f}, max={max_val:.4f}, range={dynamic_range:.4f} deg C")
    print(f"Saved figure to {args.save_path}")
    print(
        "RMSEs (deg C):",
        {
            "standard_clean": rmse_std_clean,
            "standard_perturbed": rmse_std_pert,
            "increase": rmse_std_pert - rmse_std_clean,
        },
    )


if __name__ == "__main__":
    main()
