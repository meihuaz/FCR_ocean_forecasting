import torch
import torch.nn as nn
from data.dataload_en4_profile_bg_train import GLORYDataset, GLORYDatasetTest, Denormalize
import argparse
import os
import yaml
from utils.metric import MAELoss, RMSE_metric
import time
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from model.model_factory import build_model, describe_model_type, extract_prediction

device = torch.device("cuda:0")


class Trainer():

    def count_parameters(self):
        return sum(p.numel() for p in self.model.parameters()
                   if p.requires_grad)

    def __init__(self, params):
        self.params = params

        seed = self.params.get('seed', 123)
        torch.manual_seed(seed)

        # 初始化dataloader
        glory_path_train = "/mnt/data/zhishuai/Data/train_nw_pacific/"
        glory_path_test = "/mnt/data/zhishuai/Data/test_nw_pacific/"

        batch_size = self.params.get('batch_size', 4)
        num_workers = self.params.get('num_workers', 4)
        train_shuffle = self.params.get('train_shuffle', False)
        test_shuffle = self.params.get('test_shuffle', False)
        pin_memory = self.params.get('pin_memory', True)
        persistent_workers = self.params.get('persistent_workers',
                                             num_workers > 0)
        prefetch_factor = self.params.get('prefetch_factor', 2)

        loader_kwargs = {
            'batch_size': batch_size,
            'num_workers': num_workers,
            'pin_memory': pin_memory,
        }
        if num_workers > 0:
            loader_kwargs['persistent_workers'] = persistent_workers
            loader_kwargs['prefetch_factor'] = prefetch_factor

        dataset = GLORYDataset(glory_path_train)
        self.dataloader = torch.utils.data.DataLoader(dataset,
                                                      shuffle=train_shuffle,
                                                      **loader_kwargs)

        dataset = GLORYDatasetTest(glory_path_test)
        self.dataloader_test = torch.utils.data.DataLoader(
            dataset, shuffle=test_shuffle, **loader_kwargs)

        self.write_to_log('Length of train dataset: ' +
                          str(len(self.dataloader)))
        self.write_to_log('Length of test dataset: ' +
                          str(len(self.dataloader_test)))

        self.model = build_model(self.params)
        self.write_to_log(
            f"Using {describe_model_type(self.params.get('model_type', 'swin'))} model")

        self.model.to(device)

        self.iters = 0
        self.startEpoch = 0
        self.max_epochs = self.params['max_epochs']
        self.best_valid_loss = 1.e6
        cur_epoch = 0

        # 初始化优化器和策略
        self.optimizer = torch.optim.AdamW(self.model.parameters(),
                                           lr=self.params['lr'])

        if self.params['scheduler'] == 'ReduceLROnPlateau':
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, factor=0.2, patience=5, mode='min')
        elif self.params['scheduler'] == 'CosineAnnealingLR':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                eta_min=1e-6,
                T_max=self.params['max_epochs'],
                last_epoch=-1)
        else:
            self.scheduler = None

        ckpt = torch.load(params['best_checkpoint_path'])
        self.model.load_state_dict(ckpt['model_state'] if 'model_state' in ckpt else ckpt)
        if 'optimizer_state_dict' in ckpt:
            try:
                self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            except ValueError as exc:
                self.write_to_log(f"Skipping optimizer state load for inference: {exc}")
        self.write_to_log("%s's previous weights loaded." %
                            params['load_model'])
        cur_epoch = ckpt['epoch']
        self.epoch = cur_epoch

        # 定义损失函数
        self.loss = MAELoss()
        self.l1_loss, self.l2_loss, self.bce_loss = nn.SmoothL1Loss().to(
            device), nn.MSELoss().to(device), nn.BCELoss().to(device)

        # 打印训练参数
        self.write_to_log("Number of trainable model parameters: {}".format(
            self.count_parameters()))

   

    def validate_one_epoch(self):
        valid_loss_fine_list = []

        self.model.eval()

        for idx, (test_input_temp) in enumerate(self.dataloader_test):

            glory_batch, gt, min_val, max_val = test_input_temp

            bg_field = glory_batch[:, :, :, :self.params['img_size'][2]]
            bg_field[torch.where(torch.isnan(bg_field))] = 0
            bg_field = bg_field.float().to(device)

            # 使用 interpolate 调整大小
            # 先将 [B, C, H, W, D] reshape 为 [B, C, D, H, W]，然后进行 3D resize 操作
            bg_field = bg_field.permute(0, 3, 1,
                                        2)  # 将 D 移到第三维，变为 [B, C, D, H, W]
            bg_field = F.interpolate(bg_field,
                                     size=(self.params['img_size'][0],
                                           self.params['img_size'][1]),
                                     mode='nearest',
                                     align_corners=None)
            bg_field = bg_field.float().to(device)

            gt = gt[:, :, :, :self.params['img_size'][2]]
            # gt[torch.where(torch.isnan(gt))] = 0
            gt = gt.float().to(device)

            # 使用 interpolate 调整大小
            # 先将 [B, C, H, W, D] reshape 为 [B, C, D, H, W]，然后进行 3D resize 操作
            gt = gt.permute(0, 3, 1, 2)  # 将 D 移到第三维，变为 [B, C, D, H, W]
            gt = F.interpolate(gt,
                               size=(self.params['img_size'][0],
                                     self.params['img_size'][1]),
                               mode='nearest',
                               align_corners=None)
            # 将调整大小后的 Tensor permute 回 [B, C, H1, W1, D1]
            gt = gt.permute(0, 2, 3, 1)
            gt = gt.float().to(device)

            with torch.no_grad():
                prediction = extract_prediction(self.model(bg_field))
                prediction = prediction.permute(0, 2, 3, 1)

            prediction_denormalized = Denormalize(prediction, min_val.to(device), max_val.to(device))
            gt_denormalized = Denormalize(gt, min_val.to(device), max_val.to(device))

            loss = RMSE_metric(prediction_denormalized, gt_denormalized)

            print('Validation: [{}/{}]  prediction: {}'.format(
                idx, len(self.dataloader_test), loss))

            valid_loss_fine_list.append(loss.cpu().detach().numpy())


        valid_loss_fine_average = np.sum(valid_loss_fine_list) / len(
            valid_loss_fine_list)
        log1 = {'valid_loss': valid_loss_fine_average}
        self.write_to_log(log1)

        return log1

    def save_checkpoint(self, checkpoint_path, model=None):
        """ We intentionally require a checkpoint_dir to be passed
        in order to allow Ray Tune to use this function """

        model = self.model

        torch.save(
            {
                'epoch': self.epoch,
                'model_state': model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict()
            }, checkpoint_path)

    def restore_checkpoint(self, checkpoint_path):
        """ We intentionally require a checkpoint_dir to be passed
        in order to allow Ray Tune to use this function """
        checkpoint = torch.load(checkpoint_path)
        self.model.load_state_dict(checkpoint['model_state'])
        self.startEpoch = checkpoint['epoch']
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    def write_to_log(self, text):
        if self.params['log_file'] is not None:
            print(text, file=self.params['log_file'])
        print(text)

    def write_to_log_valid(self, text):
        if self.params['valid_log_file'] is not None:
            print(text, file=self.params['valid_log_file'])
        print(text)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--general_config_in_yaml",
                        default='full_field',
                        type=str)
    parser.add_argument("--yaml_config",
                        default="./config/en4_Transformer_bg_model3.yaml",
                        type=str)
    parser.add_argument("--checkpoint_path",
                        default="",
                        type=str)
    args = parser.parse_args()

    with open(args.yaml_config) as f:
        params = yaml.safe_load(f)

    expDir = os.path.join(params['exp_dir'],
                          params['model_type'] + '_' + str(params['run_num']))
    if not os.path.exists(expDir):
        os.makedirs(expDir, exist_ok=True)
        os.makedirs(os.path.join(expDir, 'training_checkpoints/'),
                    exist_ok=True)
    log_path = os.path.join(expDir, 'log.txt')
    params['log_file'] = open(log_path, 'a', buffering=1)
    valid_log_path = os.path.join(expDir, 'valid_log.txt')
    params['valid_log_file'] = open(valid_log_path, 'a', buffering=1)

    params['checkpoint_path'] = os.path.join(expDir, 'training_checkpoints/')
    if args.checkpoint_path:
        params['best_checkpoint_path'] = args.checkpoint_path
    elif params.get('load_model_path'):
        params['best_checkpoint_path'] = params['load_model_path']
    else:
        params['best_checkpoint_path'] = "/mnt/data/zhishuai/code/best_ckpt_continue.tar"

    trainer = Trainer(params)
    trainer.validate_one_epoch()
