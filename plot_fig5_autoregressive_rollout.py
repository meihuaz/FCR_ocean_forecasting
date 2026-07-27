import argparse
import csv
import glob
import os
from datetime import datetime, timedelta
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import rcParams
import numpy as np
import torch
import torch.nn.functional as F
import yaml

from adversarial_generation import PGDAttack
from model.model_factory import build_model, extract_prediction
from utils.metric import MAELoss


REPO_ROOT = Path(__file__).resolve().parent
DEG_C = "\N{DEGREE SIGN}C"

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
rcParams["legend.fontsize"] = 15


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate the two-panel Fig. 5 autoregressive rollout plot."
    )
    parser.add_argument(
        "--yaml_config",
        default=str(REPO_ROOT / "config" / "en4_Transformer_bg_model3.yaml"),
    )
    parser.add_argument(
        "--standard_checkpoint",
        default=str(REPO_ROOT / "ckpt" / "swinunet" / "best_ckpt.tar"),
    )
    parser.add_argument(
        "--fcr_checkpoint",
        default=str(
            REPO_ROOT
            / "outputs"
            / "run_en4_bg_model3_pre_pgdTrue_simsiam_eps0.0005_lambda0.6"
            / "training_checkpoints"
            / "ckpt_51.tar"
        ),
    )
    parser.add_argument("--glory_path", default="/mnt/f/zhishuai/Data/test_nw_pacific/")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epsilon", type=float, default=5e-4)
    parser.add_argument("--alpha", type=float, default=1e-4)
    parser.add_argument("--pgd_steps", type=int, default=10)
    parser.add_argument("--rollout_steps", type=int, default=7)
    parser.add_argument("--lead_days", type=int, default=7)
    parser.add_argument("--case_count", type=int, default=20)
    parser.add_argument(
        "--case_strategy",
        choices=["first", "spread"],
        default="spread",
        help="Use the first valid chains or spread cases across the test period.",
    )
    parser.add_argument("--case_dates", nargs="*", default=None)
    parser.add_argument(
        "--output",
        default=str(
            REPO_ROOT
            / "sn-fcr_bundle_20260707"
            / "latex"
            / "figures"
            / "autoregressive_test"
            / "autoregressive_error_eps0.0005_two_panel.pdf"
        ),
    )
    parser.add_argument(
        "--csv_output",
        default=str(
            REPO_ROOT
            / "sn-fcr_bundle_20260707"
            / "latex"
            / "figures"
            / "autoregressive_test"
            / "autoregressive_error_eps0.0005_two_panel.csv"
        ),
    )
    parser.add_argument(
        "--npz_output",
        default=str(
            REPO_ROOT
            / "sn-fcr_bundle_20260707"
            / "latex"
            / "figures"
            / "autoregressive_test"
            / "autoregressive_error_eps0.0005_two_panel.npz"
        ),
    )
    return parser.parse_args()


def build_date_index(glory_path):
    by_date = {}
    for path in glob.glob(os.path.join(glory_path, "*")):
        if not os.path.isfile(path):
            continue
        name = os.path.basename(path)
        try:
            date_text = name.split("_")[-2]
            datetime.strptime(date_text, "%Y%m%d")
        except (IndexError, ValueError):
            continue
        by_date[date_text] = path
    return by_date


def shift_date(date_text, days):
    return (datetime.strptime(date_text, "%Y%m%d") + timedelta(days=days)).strftime("%Y%m%d")


def valid_initial_dates(by_date, rollout_steps, lead_days):
    dates = []
    for init_date in sorted(by_date):
        if all(shift_date(init_date, lead_days * step) in by_date for step in range(1, rollout_steps + 1)):
            dates.append(init_date)
    return dates


def select_case_dates(args, by_date):
    valid_dates = valid_initial_dates(by_date, args.rollout_steps, args.lead_days)
    if args.case_dates:
        requested = [d.replace("-", "") for d in args.case_dates]
        missing = [d for d in requested if d not in valid_dates]
        if missing:
            raise ValueError(f"Requested case dates do not have full rollout truth fields: {missing}")
        return requested

    if args.case_count > len(valid_dates):
        raise ValueError(f"Requested {args.case_count} cases but only {len(valid_dates)} valid chains exist.")

    if args.case_strategy == "spread" and args.case_count > 1:
        indices = np.linspace(0, len(valid_dates) - 1, args.case_count).round().astype(int)
        return [valid_dates[i] for i in indices]
    return valid_dates[: args.case_count]


def load_model(yaml_config, checkpoint_path, device):
    with open(yaml_config) as f:
        params = yaml.safe_load(f)
    model = build_model(params)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state = checkpoint["model_state"] if isinstance(checkpoint, dict) and "model_state" in checkpoint else checkpoint
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model, params


def load_normalized_initial_and_truths(by_date, init_date, rollout_steps, lead_days, img_size, device):
    init_raw = np.load(by_date[init_date])
    min_value = float(np.nanmin(init_raw))
    max_value = float(np.nanmax(init_raw))
    scale = max_value - min_value
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError(f"Invalid normalization range for {init_date}: {scale}")

    init_norm = (init_raw - min_value) / scale
    init_norm[np.isnan(init_norm)] = 0.0
    init_tensor = torch.from_numpy(init_norm[:, :, : img_size[2]]).unsqueeze(0).float()
    init_tensor = init_tensor.permute(0, 3, 1, 2).to(device)
    init_tensor = F.interpolate(init_tensor, size=(img_size[0], img_size[1]), mode="nearest")

    truth_tensors = []
    mask_tensors = []
    truth_dates = []
    for step in range(1, rollout_steps + 1):
        truth_date = shift_date(init_date, lead_days * step)
        truth_raw = np.load(by_date[truth_date])
        valid_mask = np.isfinite(truth_raw[:, :, : img_size[2]])
        truth_norm = (truth_raw - min_value) / scale
        truth_norm[np.isnan(truth_norm)] = 0.0

        truth_tensor = torch.from_numpy(truth_norm[:, :, : img_size[2]]).unsqueeze(0).float()
        truth_tensor = truth_tensor.permute(0, 3, 1, 2).to(device)
        truth_tensor = F.interpolate(truth_tensor, size=(img_size[0], img_size[1]), mode="nearest")
        truth_tensor = truth_tensor.permute(0, 2, 3, 1)

        mask_tensor = torch.from_numpy(valid_mask).unsqueeze(0).float()
        mask_tensor = mask_tensor.permute(0, 3, 1, 2).to(device)
        mask_tensor = F.interpolate(mask_tensor, size=(img_size[0], img_size[1]), mode="nearest")
        mask_tensor = mask_tensor.permute(0, 2, 3, 1)

        truth_tensors.append(truth_tensor)
        mask_tensors.append(mask_tensor)
        truth_dates.append(truth_date)

    ocean_mask = mask_tensors[0].permute(0, 3, 1, 2)
    return init_tensor, truth_tensors, mask_tensors, ocean_mask, min_value, max_value, truth_dates


def masked_rmse_celsius(pred_bhwc, truth_bhwc, mask_bhwc, scale):
    diff = (pred_bhwc - truth_bhwc) * scale
    mse = (diff.square() * mask_bhwc).sum() / mask_bhwc.sum().clamp_min(1.0)
    return float(torch.sqrt(mse).detach().cpu())


def rollout_rmse(model, initial_tensor, truth_tensors, mask_tensors, ocean_mask, scale):
    rmses = []
    current = initial_tensor
    with torch.no_grad():
        for truth, mask in zip(truth_tensors, mask_tensors):
            pred_bchw = extract_prediction(model(current))
            pred_bhwc = pred_bchw.permute(0, 2, 3, 1)
            rmses.append(masked_rmse_celsius(pred_bhwc, truth, mask, scale))
            current = (pred_bchw * ocean_mask).detach()
    return np.asarray(rmses, dtype=np.float64)


def make_initial_perturbation(model, initial_tensor, first_truth, epsilon, alpha, pgd_steps, device):
    criterion = MAELoss()
    attacker = PGDAttack(
        model,
        epsilon=epsilon,
        alpha=alpha,
        num_iter=pgd_steps,
        device=device,
    )
    return attacker.generate(initial_tensor, first_truth, criterion, random_start=False)


def summarize(values):
    mean = values.mean(axis=0)
    std = values.std(axis=0, ddof=1) if values.shape[0] > 1 else np.zeros_like(mean)
    return {
        "mean": mean,
        "min": mean - std,
        "max": mean + std,
    }


def save_csv(path, lead_times, case_dates, arrays):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["case_date", "lead_days", "series", "rmse_c"])
        for series_name, values in arrays.items():
            for case_date, row in zip(case_dates, values):
                for lead, rmse_value in zip(lead_times, row):
                    writer.writerow([case_date, lead, series_name, f"{rmse_value:.8f}"])


def plot_figure(output, lead_times, arrays):
    os.makedirs(os.path.dirname(output), exist_ok=True)
    standard_clean = arrays["standard_clean"]
    standard_pert = arrays["standard_perturbed"]
    fcr_clean = arrays["fcr_clean"]
    fcr_pert = arrays["fcr_perturbed"]
    standard_delta = standard_pert - standard_clean
    fcr_delta = fcr_pert - fcr_clean

    summaries = {
        "standard_clean": summarize(standard_clean),
        "standard_perturbed": summarize(standard_pert),
        "fcr_clean": summarize(fcr_clean),
        "fcr_perturbed": summarize(fcr_pert),
        "standard_delta": summarize(standard_delta),
        "fcr_delta": summarize(fcr_delta),
    }

    fig, axes = plt.subplots(1, 2, figsize=(16, 6.2), constrained_layout=True)
    ax = axes[0]
    specs = [
        ("standard_clean", "Standard Swin-T, clean", "#7aa6c2", "--"),
        ("standard_perturbed", "Standard Swin-T, perturbed", "#1f77b4", "-"),
        ("fcr_clean", "FCR, clean", "#f2a65a", "--"),
        ("fcr_perturbed", "FCR, perturbed", "#d95f02", "-"),
    ]
    for key, label, color, linestyle in specs:
        summary = summaries[key]
        ax.plot(lead_times, summary["mean"], label=label, color=color, linestyle=linestyle, linewidth=3.0)
        ax.fill_between(lead_times, summary["min"], summary["max"], color=color, alpha=0.12, linewidth=0)

    std_day49 = summaries["standard_perturbed"]["mean"][-1]
    fcr_day49 = summaries["fcr_perturbed"]["mean"][-1]
    reduction = (std_day49 - fcr_day49) / std_day49 * 100.0
    ax.annotate(
        f"{reduction:.1f}% lower RMSE\nat day {lead_times[-1]}",
        xy=(lead_times[-1], fcr_day49),
        xytext=(lead_times[-1] - 18, fcr_day49 + 0.33 * (std_day49 - fcr_day49)),
        arrowprops=dict(arrowstyle="->", color="black", lw=1.5),
        fontsize=15,
        ha="left",
        va="center",
    )
    ax.set_title("(a) RMSE growth")
    ax.set_xlabel("Lead time (days)")
    ax.set_ylabel(f"RMSE ({DEG_C})")
    ax.set_xticks(lead_times)
    ax.grid(True, alpha=0.25, linewidth=0.8)
    ax.legend(frameon=False, loc="upper left")

    ax = axes[1]
    for key, label, color in [
        ("standard_delta", "Standard Swin-T", "#1f77b4"),
        ("fcr_delta", "FCR", "#d95f02"),
    ]:
        summary = summaries[key]
        ax.plot(lead_times, summary["mean"], label=label, color=color, linewidth=3.0)
        ax.fill_between(lead_times, summary["min"], summary["max"], color=color, alpha=0.14, linewidth=0)
    ax.axhline(0.0, color="0.35", linewidth=1.2, linestyle=":")
    ax.set_title(r"(b) Perturbation-induced $\Delta$RMSE")
    ax.set_xlabel("Lead time (days)")
    ax.set_ylabel(r"$\Delta$RMSE (" + DEG_C + ")")
    ax.set_xticks(lead_times)
    ax.grid(True, alpha=0.25, linewidth=0.8)
    ax.legend(frameon=False, loc="upper left")

    fig.savefig(output, bbox_inches="tight")
    png_output = str(Path(output).with_suffix(".png"))
    fig.savefig(png_output, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return {
        "standard_day49": std_day49,
        "fcr_day49": fcr_day49,
        "reduction_pct": reduction,
        "png_output": png_output,
    }


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    by_date = build_date_index(args.glory_path)
    case_dates = select_case_dates(args, by_date)
    print(f"Selected rollout initial dates: {', '.join(case_dates)}")

    standard_model, params = load_model(args.yaml_config, args.standard_checkpoint, device)
    fcr_model, _ = load_model(args.yaml_config, args.fcr_checkpoint, device)
    img_size = params["img_size"]

    arrays = {
        "standard_clean": [],
        "standard_perturbed": [],
        "fcr_clean": [],
        "fcr_perturbed": [],
    }

    for idx, init_date in enumerate(case_dates, start=1):
        print(f"[{idx}/{len(case_dates)}] rollout from {init_date}")
        initial_tensor, truth_tensors, mask_tensors, ocean_mask, min_value, max_value, truth_dates = (
            load_normalized_initial_and_truths(
                by_date,
                init_date,
                args.rollout_steps,
                args.lead_days,
                img_size,
                device,
            )
        )
        scale = max_value - min_value
        perturbed_initial = make_initial_perturbation(
            standard_model,
            initial_tensor,
            truth_tensors[0],
            args.epsilon,
            args.alpha,
            args.pgd_steps,
            device,
        )
        perturbed_initial = perturbed_initial * ocean_mask

        arrays["standard_clean"].append(
            rollout_rmse(standard_model, initial_tensor, truth_tensors, mask_tensors, ocean_mask, scale)
        )
        arrays["standard_perturbed"].append(
            rollout_rmse(standard_model, perturbed_initial, truth_tensors, mask_tensors, ocean_mask, scale)
        )
        arrays["fcr_clean"].append(
            rollout_rmse(fcr_model, initial_tensor, truth_tensors, mask_tensors, ocean_mask, scale)
        )
        arrays["fcr_perturbed"].append(
            rollout_rmse(fcr_model, perturbed_initial, truth_tensors, mask_tensors, ocean_mask, scale)
        )
        torch.cuda.empty_cache()

    arrays = {key: np.vstack(value) for key, value in arrays.items()}
    lead_times = np.arange(1, args.rollout_steps + 1) * args.lead_days

    save_csv(args.csv_output, lead_times, case_dates, arrays)
    os.makedirs(os.path.dirname(args.npz_output), exist_ok=True)
    np.savez_compressed(
        args.npz_output,
        lead_times=lead_times,
        case_dates=np.asarray(case_dates),
        **arrays,
    )
    stats = plot_figure(args.output, lead_times, arrays)

    print(f"Saved figure to {args.output}")
    print(f"Saved PNG to {stats['png_output']}")
    print(f"Saved CSV to {args.csv_output}")
    print(f"Saved NPZ to {args.npz_output}")
    print(
        "Day-49 perturbed RMSE:",
        f"Standard={stats['standard_day49']:.3f} {DEG_C},",
        f"FCR={stats['fcr_day49']:.3f} {DEG_C},",
        f"reduction={stats['reduction_pct']:.1f}%",
    )


if __name__ == "__main__":
    main()
