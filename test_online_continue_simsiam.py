import os
import argparse
from typing import Tuple

import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from model.swinunet import SwinTransformer
from data.dataload_en4_profile_bg_train import GLORYDataset, GLORYDatasetTest
from utils.metric import MAELoss

from torch.utils.data import DataLoader


parser = argparse.ArgumentParser()
parser.add_argument(
    "--yaml_config",
    default="./config/en4_Transformer_bg_model3.yaml",
    type=str,
    help="Path to YAML configuration file",
)
parser.add_argument(
    "--device",
    default="cuda:0",
    type=str,
    help="Device to use, e.g. 'cuda:1' or 'cpu'",
)
# 是否使用 PGD 对抗训练
parser.add_argument(
    "--use_pgd",
    action="store_false",
    help="Enable PGD adversarial training",
)
parser.add_argument(
    "--pgd_epsilon",
    type=float,
    default=5e-4,
    help="PGD max perturbation (epsilon)",
)
parser.add_argument(
    "--pgd_alpha",
    type=float,
    default=1e-4,
    help="PGD step size (alpha)",
)
parser.add_argument(
    "--pgd_steps",
    type=int,
    default=10,
    help="Number of PGD iterations",
)
parser.add_argument(
    "--adv_lambda",
    type=float,
    default=0.6,
    help="Weight λ for adversarial loss: loss = L_clean + λ * L_adv",
)
# 新增参数：是否使用 TRADES
parser.add_argument(
    "--use_trades",
    type=bool,
    default=False,
    help="Enable TRADES adversarial training",
)
# 新增参数：SimSiam 权重
parser.add_argument(
    "--simsiam_lambda",
    type=float,
    default=1.0,
    help="Weight for SimSiam consistency loss",
)
# 新增参数：用于区分不同实验的目录后缀
parser.add_argument(
    "--run_suffix",
    type=str,
    default="",
    help="Suffix for the experiment directory to distinguish tuning runs",
)
parser.add_argument(
    "--exp_dir",
    type=str,
    default=None,
    help="Base directory for logs/checkpoints. Falls back to ./exp/PoinTr if the YAML path is not writable.",
)
parser.add_argument("--continue_training",
                    action="store_false",
                    help="Continue training from existing checkpoints")
parser.add_argument("--ckpt_path",
                    type=str,
                    default="/mnt/e/project/train_online_continue_simsiam_minimal/outputs/run_en4_bg_model3_pre_pgdTrue_simsiam_eps0.0005_lambda0.6/training_checkpoints/ckpt_51.tar",
                    help="Path to existing checkpoint to continue training")

args = parser.parse_args()


# -----------------------------
# SimSiam 模块
# -----------------------------
class SimSiam(nn.Module):
    """
    SimSiam for dense features (B, C, H, W) using 1x1 Convs (equivalent to pixel-wise MLP)
    """
    def __init__(self, in_dim, dim=256, pred_dim=64):
        super().__init__()
        
        # Projector: 3-layer MLP (1x1 Conv)
        self.projector = nn.Sequential(
            nn.Conv2d(in_dim, dim, 1, bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim, 1, bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim, 1, bias=False),
            nn.BatchNorm2d(dim, affine=False) 
        )
        
        # Predictor: 2-layer MLP (1x1 Conv)
        self.predictor = nn.Sequential(
            nn.Conv2d(dim, pred_dim, 1, bias=False),
            nn.BatchNorm2d(pred_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(pred_dim, dim, 1)
        )

    def forward(self, x1, x2):
        """
        x1, x2: Features from encoder (B, C, H, W)
        """
        z1 = self.projector(x1)
        z2 = self.projector(x2)
        
        p1 = self.predictor(z1)
        p2 = self.predictor(z2)
        
        return p1, p2, z1.detach(), z2.detach()

def simsiam_loss_func(p, z):
    # Negative Cosine Similarity
    return - F.cosine_similarity(p, z, dim=1).mean()


# -----------------------------
# PGD 对抗攻击（在线生成对抗样本）
# -----------------------------
class PGDAttack:
    def __init__(
            self,
            model: nn.Module,
            epsilon: float = 0.1,
            alpha: float = 0.01,
            num_iter: int = 10,
            norm: str = "inf",
            device: torch.device = torch.device("cuda:1"),
    ):
        self.model = model
        self.epsilon = epsilon
        self.alpha = alpha
        self.num_iter = num_iter
        self.norm = norm
        self.device = device

    def generate(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        criterion: nn.Module,
        random_start: bool = False,
    ) -> torch.Tensor:
        self.model.eval()
        x = x.to(self.device)
        y = y.to(self.device)

        if random_start:
            x_adv = x + torch.empty_like(x).uniform_(-self.epsilon, self.epsilon)
            x_adv = torch.clamp(x_adv, x.min(), x.max())
        else:
            x_adv = x.clone().detach()

        x_adv = x_adv.detach()

        for _ in range(self.num_iter):
            x_adv.requires_grad = True
            # 注意：这里只用 output 进行攻击，不涉及 feature
            _, output = self.model(x_adv)
            output = output.permute(0, 2, 3, 1)
            loss = criterion(output, y)

            self.model.zero_grad()
            loss.backward()

            with torch.no_grad():
                grad = x_adv.grad
                if self.norm == "inf":
                    perturbation = self.alpha * grad.sign()
                else:
                    perturbation = torch.zeros_like(grad)

                x_adv = x_adv + perturbation
                delta = x_adv - x
                if self.norm == "inf":
                    delta = torch.clamp(delta, -self.epsilon, self.epsilon)

                x_adv = x + delta
                x_adv = torch.clamp(x_adv, x.min(), x.max())

        return x_adv.detach()


# -----------------------------
# 对抗训练 Trainer (集成 SimSiam)
# -----------------------------
class AdversarialTrainer:
    def __init__(
        self,
        params: dict,
        device: str = "cuda:1",
        use_pgd: bool = False,
        use_trades: bool = False,
        pgd_epsilon: float = 0.002,
        pgd_alpha: float = 0.01,
        pgd_steps: int = 10,
        adv_lambda: float = 1.0,
        simsiam_lambda: float = 1.0,
    ):
        self.params = params
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        self.pgd_epsilon = pgd_epsilon
        self.pgd_alpha = pgd_alpha
        self.pgd_steps = pgd_steps

        self.use_pgd = use_pgd and (adv_lambda > 0.0)
        self.use_trades = use_trades
        self.adv_lambda = adv_lambda if self.use_pgd else 0.0
        self.simsiam_lambda = simsiam_lambda

        seed = self.params.get("seed", 123)
        torch.manual_seed(seed)
        np.random.seed(seed)

        self._init_dataloaders()

        # 模型
        self.model = SwinTransformer(self.params).to(self.device)
        
        # SimSiam 模块
        # SwinTransformer 的 embed_dim 是 192 (根据 config)
        # layer3 output has 2 * embed_dim channels
        embed_dim = self.params['net_image'].get('embed_dim', 192)
        self.simsiam = SimSiam(in_dim=embed_dim * 2).to(self.device)

        self.epoch = 0
        self.max_epochs = self.params["max_epochs"]
        self.best_valid_loss = 1.0e6


        ckpt = torch.load(args.ckpt_path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state"])
        if "simsiam_state" in ckpt:
            self.simsiam.load_state_dict(ckpt["simsiam_state"])
        self.write_to_log("Loaded pretrained model from " + args.ckpt_path)


 
        self.loss_fn = MAELoss()

        if self.use_pgd:
            self.pgd_attack = PGDAttack(
                self.model,
                epsilon=pgd_epsilon,
                alpha=pgd_alpha,
                num_iter=pgd_steps,
                device=self.device,
            )
            mode_str = "TRADES" if self.use_trades else "Standard PGD"
            self.write_to_log(
                f"*** Using {mode_str} + SimSiam adversarial training: "
                f"eps={pgd_epsilon}, alpha={pgd_alpha}, steps={pgd_steps}, "
                f"adv_lambda={self.adv_lambda}, simsiam_lambda={self.simsiam_lambda} ***")
        else:
            self.pgd_attack = None
            self.write_to_log("*** Training with clean samples only (no PGD) ***")

    def _init_dataloaders(self):
        glory_path_train = "/mnt/data/zhishuai/Data/train_nw_pacific/"
        glory_path_test = "/mnt/data/zhishuai/Data/test_nw_pacific/"
        train_dataset = GLORYDataset(glory_path_train)
        test_dataset = GLORYDatasetTest(glory_path_test)

        batch_size = self.params.get("batch_size", 2)
        num_workers = self.params.get("num_workers", 2)

        self.dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
        self.dataloader_test = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
        self.write_to_log(f"Train dataset length (batches): {len(self.dataloader)}")
        self.write_to_log(f"Validation dataset length (batches): {len(self.dataloader_test)}")

    def preprocess_data(self, glory_batch, gt) -> Tuple[torch.Tensor, torch.Tensor]:
        bg_field = glory_batch[:, :, :, :self.params["img_size"][2]]
        bg_field[torch.isnan(bg_field)] = 0
        bg_field = bg_field.float().to(self.device)
        bg_field = bg_field.permute(0, 3, 1, 2)
        bg_field = F.interpolate(bg_field, size=(self.params["img_size"][0], self.params["img_size"][1]), mode="nearest")

        gt = gt[:, :, :, :self.params["img_size"][2]]
        gt[torch.isnan(gt)] = 0
        gt = gt.float().to(self.device)
        gt = gt.permute(0, 3, 1, 2)
        gt = F.interpolate(gt, size=(self.params["img_size"][0], self.params["img_size"][1]), mode="nearest")
        gt = gt.permute(0, 2, 3, 1)
        return bg_field, gt

    def train_one_epoch(self):
        self.model.train()
        self.simsiam.train()
        total_loss = 0.0
        num_batches = len(self.dataloader)

        pbar = tqdm(self.dataloader, desc=f"Epoch {self.epoch}/{self.max_epochs}")

        for i_batch, batch_data in enumerate(pbar):
            if len(batch_data) == 2:
                glory_batch, gt = batch_data
            else:
                glory_batch, gt, *_ = batch_data

            bg_field, gt = self.preprocess_data(glory_batch, gt)

            self.optimizer.zero_grad()

            # 1. Clean Forward
            feature_clean, pred_clean = self.model(bg_field)
            pred_clean = pred_clean.permute(0, 2, 3, 1)
            loss_clean = self.loss_fn(pred_clean, gt)

            loss = loss_clean
            loss_info = {"clean": loss_clean.item()}

            # 2. PGD Adversarial Training + SimSiam
            if self.use_pgd and self.pgd_attack is not None:
                # 生成对抗样本
                x_adv = self.pgd_attack.generate(bg_field, gt, self.loss_fn, random_start=False)

                # 对抗样本前向
                feature_adv, pred_adv = self.model(x_adv)
                pred_adv = pred_adv.permute(0, 2, 3, 1)

                # 对抗损失
                if self.use_trades:
                    loss_adv = self.loss_fn(pred_adv, pred_clean.detach()) # TRADES通常detach clean target
                else:
                    loss_adv = self.loss_fn(pred_adv, gt)
                
                loss_info["adv"] = loss_adv.item()

                # SimSiam Consistency Loss
                # 强制 clean 和 adv 的特征表示一致
                p1, p2, z1, z2 = self.simsiam(feature_clean, feature_adv)
                
                # 修改为单向约束：让 adv 的特征去逼近 clean 的特征
                # 避免 clean 的特征被 adv (噪声) 带偏，导致 clean 性能下降
                # 只使用 p2 (adv prediction) 逼近 z1 (clean projection)
                loss_simsiam = simsiam_loss_func(p2, z1)
                
                loss_info["simsiam"] = loss_simsiam.item()

                # 总损失
                loss = loss_clean + self.adv_lambda * loss_adv + self.simsiam_lambda * loss_simsiam

            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

            if "adv" in loss_info:
                pbar.set_postfix({
                    "loss": f"{loss.item():.4f}",
                    "clean": f"{loss_info['clean']:.4f}",
                    "adv": f"{loss_info['adv']:.4f}",
                    "sim": f"{loss_info['simsiam']:.4f}",
                })
            else:
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})

            if i_batch % 50 == 0:
                if "adv" in loss_info:
                    log_msg = (f"[{self.epoch}/{self.max_epochs}] [{i_batch}/{num_batches}] "
                               f"loss={loss.item():.4f}, clean={loss_info['clean']:.4f}, "
                               f"adv={loss_info['adv']:.4f}, sim={loss_info['simsiam']:.4f}")
                else:
                    log_msg = (f"[{self.epoch}/{self.max_epochs}] [{i_batch}/{num_batches}] "
                               f"loss={loss.item():.4f}")
                self.write_to_log(log_msg)

        avg_loss = total_loss / num_batches
        return {"train_loss": avg_loss}

    def validate_one_epoch(self):
        self.model.eval()
        valid_loss_list = []
        valid_adv_loss_list = []

        if self.pgd_attack is not None:
            val_attacker = self.pgd_attack
        else:
            val_attacker = PGDAttack(self.model, epsilon=self.pgd_epsilon, alpha=self.pgd_alpha, num_iter=self.pgd_steps, device=self.device)

        for batch_data in tqdm(self.dataloader_test, desc="Validation"):
            if len(batch_data) == 2:
                glory_batch, gt = batch_data
            else:
                glory_batch, gt, *_ = batch_data

            bg_field, gt = self.preprocess_data(glory_batch, gt)

            with torch.no_grad():
                _, prediction = self.model(bg_field)
                prediction = prediction.permute(0, 2, 3, 1)
                loss_clean = self.loss_fn(prediction, gt)
                valid_loss_list.append(loss_clean.item())

            # Validation on Adversarial Samples
            # We need gradients to generate adversarial samples, so we enable grad temporarily for the attack
            with torch.enable_grad():
                x_adv = val_attacker.generate(bg_field, gt, self.loss_fn, random_start=False)
            
            with torch.no_grad():
                _, pred_adv = self.model(x_adv)
                pred_adv = pred_adv.permute(0, 2, 3, 1)
                loss_adv = self.loss_fn(pred_adv, gt)
                valid_adv_loss_list.append(loss_adv.item())

        avg_valid_loss = float(np.mean(valid_loss_list))
        avg_valid_adv_loss = float(np.mean(valid_adv_loss_list))
        self.write_to_log(f"Validation Clean Loss: {avg_valid_loss:.4f}")
        self.write_to_log(f"Validation Adv Loss: {avg_valid_adv_loss:.4f}")

        return {"valid_loss": avg_valid_loss, "valid_adv_loss": avg_valid_adv_loss}


    def write_to_log(self, text: str):
        log_file = self.params.get("log_file")
        if log_file is not None:
            print(text, file=log_file)
        print(text)



def main():
    with open(args.yaml_config) as f:
        params = yaml.safe_load(f)

    trades_str = "_trades" if args.use_trades else ""
    run_name = f"run_{params['run_num']}_pgd{args.use_pgd}{trades_str}_simsiam_eps{args.pgd_epsilon}_lambda{args.adv_lambda}{args.run_suffix}"
    configured_exp_dir = args.exp_dir or params.get("exp_dir", "./exp/PoinTr")
    exp_dir = os.path.join(configured_exp_dir, run_name)
    try:
        os.makedirs(exp_dir, exist_ok=True)
    except PermissionError:
        fallback_exp_dir = os.path.join(os.getcwd(), "exp", "PoinTr")
        exp_dir = os.path.join(fallback_exp_dir, run_name)
        print(
            f"Configured exp_dir is not writable: {configured_exp_dir}. "
            f"Using local exp_dir instead: {fallback_exp_dir}"
        )
        os.makedirs(exp_dir, exist_ok=True)
    ckpt_dir = os.path.join(exp_dir, "training_checkpoints/")
    os.makedirs(ckpt_dir, exist_ok=True)

    log_path = os.path.join(exp_dir, "train_log.txt")
    print(f"Logging to {log_path}")
    params["log_file"] = open(log_path, "a", buffering=1)
    params["checkpoint_path"] = ckpt_dir

    trainer = AdversarialTrainer(
        params=params,
        device=args.device,
        use_pgd=args.use_pgd,
        use_trades=args.use_trades,
        pgd_epsilon=args.pgd_epsilon,
        pgd_alpha=args.pgd_alpha,
        pgd_steps=args.pgd_steps,
        adv_lambda=args.adv_lambda,
        simsiam_lambda=args.simsiam_lambda,
    )

    trainer.validate_one_epoch()

    if params.get("log_file") is not None:
        params["log_file"].close()

if __name__ == "__main__":
    main()
