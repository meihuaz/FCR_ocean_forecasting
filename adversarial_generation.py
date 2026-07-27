import os
import argparse
from typing import Tuple, Dict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml

from data.dataload_en4_profile_bg_samp_gen import GLORYDataset, GLORYDatasetTest
from model.model_factory import build_model, extract_prediction


class PGDAttack:
    """
    PGD (Projected Gradient Descent) 对抗攻击类
    用于生成海洋温盐场预报的对抗样本
    """

    def __init__(
        self,
        model: nn.Module,
        epsilon: float = 0.1,
        alpha: float = 0.01,
        num_iter: int = 10,
        norm: str = "inf",
        device: str = "cuda",
    ):
        """
        Args:
            model: 预报模型
            epsilon: 最大扰动幅度（初始误差上界）
            alpha: 每次迭代的步长
            num_iter: PGD迭代次数
            norm: 范数类型 ('inf', '2')
            device: 计算设备
        """
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
        """
        生成对抗样本

        Args:
            x: 输入数据 (batch_size, channels, height, width)
            y: 目标标签/真实值
            criterion: 损失函数
            random_start: 是否随机初始化扰动

        Returns:
            对抗样本张量，与 x 形状一致
        """
        x = x.to(self.device)
        y = y.to(self.device)

        # 初始化对抗样本
        x_adv = x.clone().detach()
        if random_start:
            if self.norm == "inf":
                x_adv = x_adv + torch.empty_like(x_adv).uniform_(
                    -self.epsilon, self.epsilon
                )
                delta = torch.clamp(x_adv - x, -self.epsilon, self.epsilon)
                x_adv = torch.clamp(x + delta, x.min(), x.max())
            else:
                raise NotImplementedError(
                    f"random_start is only implemented for norm={self.norm!r}"
                )

        # PGD迭代
        for _ in range(self.num_iter):
            x_adv.requires_grad = True

            # 前向传播
            output = extract_prediction(self.model(x_adv))
            loss = criterion(output.permute(0, 2, 3, 1), y)

            # print('loss: ', loss.item())

            # 反向传播获取梯度
            self.model.zero_grad()
            loss.backward()

            with torch.no_grad():
                grad = x_adv.grad

                if self.norm == "inf":
                    # L-inf 范数约束
                    perturbation = self.alpha * grad.sign()
                else:
                    # 其他范数暂未实现，保持原行为：不做更新
                    perturbation = torch.zeros_like(grad)

                # 更新对抗样本
                x_adv = x_adv + perturbation

                # 投影到扰动范围内
                delta = x_adv - x
                if self.norm == "inf":
                    delta = torch.clamp(delta, -self.epsilon, self.epsilon)

                x_adv = x + delta

                # 确保数据在有效范围内（这里用 x 的 min/max 作为边界）
                x_adv = torch.clamp(x_adv, x.min(), x.max())

        return x_adv.detach()


class AdversarialDataset:
    """对抗样本数据集管理"""

    def __init__(self, save_dir: str = "./adversarial_samples_train"):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

    def save_batch(
        self,
        x_clean: torch.Tensor,
        x_adv: torch.Tensor,
        y: torch.Tensor,
        batch_idx: int,
    ):
        """保存一批对抗样本"""
        save_path = os.path.join(self.save_dir, f"batch_{batch_idx}.npz")

        data_dict = {
            "x_clean": x_clean.cpu().numpy(),
            "x_adv": x_adv.cpu().numpy(),
            "y": y.cpu().numpy(),
        }

        np.savez_compressed(save_path, **data_dict)

    def load_batch(self, batch_idx: int) -> Dict[str, np.ndarray]:
        """加载一批对抗样本"""
        load_path = os.path.join(self.save_dir, f"batch_{batch_idx}.npz")
        return dict(np.load(load_path))


def generate_adversarial_dataset(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    epsilon: float = 0.1,
    alpha: float = 0.01,
    num_iter: int = 10,
    save_dir: str = "./adversarial_samples_train",
    device: str = "cuda",
) -> None:
    """
    批量生成对抗样本数据集并保存为 npz 文件

    Args:
        model: 预报模型
        dataloader: 数据加载器
        criterion: 损失函数
        epsilon: 最大扰动幅度
        alpha: PGD步长
        num_iter: PGD迭代次数
        save_dir: 保存路径
        device: 计算设备
    """
    model.eval()
    pgd_attack = PGDAttack(model, epsilon, alpha, num_iter, device=device)
    adv_dataset = AdversarialDataset(save_dir)

    for batch_idx, (x, y, *_) in enumerate(tqdm(dataloader)):
        # 原始数据为 (B, H, W, C)，与模型输入保持一致
        x_chw = x.permute(0, 3, 1, 2)
        x_adv = pgd_attack.generate(x_chw, y, criterion)
        print(criterion(extract_prediction(model(x_adv)).permute(0, 2, 3, 1), y.cuda()).item())
        print(criterion(extract_prediction(model(x_chw.cuda())).permute(0, 2, 3, 1), y.cuda()).item())
        # adv_dataset.save_batch(x_chw, x_adv, y, batch_idx)


class load_model:
    """
    仅用于加载 SwinTransformer 权重和数据加载器
    用于推理 / 对抗攻击的简单封装
    """

    def __init__(self, params, device: str = "cuda:0"):
        self.params = params
        self.device = torch.device(device)

        # 初始化模型
        self.model = build_model(self.params).to(self.device)

        # 如果需要，加载 checkpoint 权重
        if self.params.get("load_model", False):
            ckpt_path = self.params["load_model_path"]
            ckpt = torch.load(ckpt_path, map_location=self.device)
            self.model.load_state_dict(ckpt["model_state"])
            print(f"Loaded pretrained weights from {ckpt_path}")
        else:
            print(
                "No pretrained checkpoint specified, using randomly initialized model."
            )

        # 推理模式
        self.model.eval()

        glory_path_train = self.params.get(
            "glory_path_train", "/mnt/data/zhishuai/Data/train_nw_pacific/")
        glory_path_test = self.params.get(
            "glory_path_test", "/mnt/data/zhishuai/Data/test_nw_pacific/")

        batch_size = self.params.get("batch_size", 1)
        num_workers = self.params.get("num_workers", 1)

        # 训练集
        train_dataset = GLORYDataset(glory_path_train)
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            prefetch_factor=2,
            persistent_workers=True,
        )

        # 测试集 / 验证集
        test_dataset = GLORYDatasetTest(glory_path_test)
        self.test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            prefetch_factor=2,
            persistent_workers=True,
        )

        print(f"Train dataloader length: {len(self.train_loader)}")
        print(f"Test dataloader length: {len(self.test_loader)}")

    def get_model(self) -> nn.Module:
        """返回已经加载好权重的模型，用于推理/对抗攻击等"""
        return self.model

    def get_dataloaders(self):
        """返回训练和测试 DataLoader"""
        return self.train_loader, self.test_loader


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--yaml_config",
        default="./config/en4_Transformer_bg_model3.yaml",
        type=str,
    )
    parser.add_argument("--device", default="cuda:0", type=str)
    parser.add_argument("--epsilon", type=float, default=5e-4)
    parser.add_argument("--alpha", type=float, default=1e-4)
    parser.add_argument("--num_iter", type=int, default=10)
    args = parser.parse_args()

    with open(args.yaml_config) as f:
        params = yaml.safe_load(f)

    model_loader = load_model(params, device=args.device)
    model_for_inference = model_loader.get_model()
    train_loader, test_loader = model_loader.get_dataloaders()

    criterion = nn.L1Loss()

    generate_adversarial_dataset(
        model_for_inference,
        train_loader,
        criterion,
        epsilon=args.epsilon,
        alpha=args.alpha,
        num_iter=args.num_iter,
        save_dir=
        f"./adv_eps{args.epsilon}_a{args.alpha}_it{args.num_iter}_train",
        device=args.device,
    )
