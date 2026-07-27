from model.swinunet import SwinTransformer
from model.resnet_baseline import ResNetBaseline
from model.convlstm_baseline import ConvLSTMBaseline
from model.fno_baseline import FNOBaseline
import torch
import torch.nn as nn
from data.dataload_en4_profile_bg_train import GLORYDataset, GLORYDatasetTest
import argparse
import os
import yaml
from utils.metric import MAELoss
import time
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F

device = torch.device("cuda:0")
torch.backends.cudnn.benchmark = True


class Trainer():

    @staticmethod
    def format_eta(seconds):
        seconds = max(0, int(seconds))
        hours, remainder = divmod(seconds, 3600)
        minutes, secs = divmod(remainder, 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"

    def count_parameters(self):
        return sum(p.numel() for p in self.model.parameters()
                   if p.requires_grad)

    def __init__(self, params):
        self.params = params

        seed = self.params.get('seed', 123)
        torch.manual_seed(seed)

        # 初始化dataloader
        glory_path_train = "/mnt/f/zhishuai/Data/train_nw_pacific/"
        glory_path_test = "/mnt/f/zhishuai/Data/test_nw_pacific/"

        batch_size = self.params.get('batch_size', 4)
        num_workers = self.params.get('num_workers', 4)
        train_shuffle = self.params.get('train_shuffle', True)
        test_shuffle = self.params.get('test_shuffle', False)
        pin_memory = self.params.get('pin_memory', True)
        persistent_workers = self.params.get('persistent_workers', num_workers
                                             > 0)
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

        # 初始化new Senseiver
        if self.params.get('model_type', 'swin') == 'resnet':
            self.model = ResNetBaseline(self.params)
            self.write_to_log("Using ResNet Baseline model")
        elif self.params.get('model_type', 'swin') == 'convlstm':
            self.model = ConvLSTMBaseline(self.params)
            self.write_to_log("Using ConvLSTM Baseline model")
        elif self.params.get('model_type', 'swin') == 'fno':
            self.model = FNOBaseline(self.params)
            self.write_to_log("Using FNO Baseline model")
        else:
            self.model = SwinTransformer(self.params)
            self.write_to_log("Using SwinTransformer model")

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

        if self.params['load_model']:
            ckpt = torch.load(params['load_model_path'])
            self.model.load_state_dict(ckpt['model_state'])
            self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
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

    def train(self):
        if self.params['log_to_screen']:
            self.write_to_log("Starting Training Loop...")

        start_epoch = self.epoch
        training_start = time.time()

        for epoch in range(start_epoch, self.max_epochs):
            epoch_start = time.time()

            # 训练阶段
            self.epoch = epoch
            self.train_one_epoch()

            # 验证阶段
            valid_logs = self.validate_one_epoch()

            if self.params['scheduler'] == 'CosineAnnealingLR':
                self.scheduler.step()

            lr = self.optimizer.param_groups[0]['lr']
            self.write_to_log(f'********** learning rate: {lr} *********')

            epoch_time = time.time() - epoch_start
            finished_epochs = epoch - start_epoch + 1
            avg_epoch_time = (time.time() - training_start) / finished_epochs
            remaining_epochs = self.max_epochs - epoch - 1
            total_eta = avg_epoch_time * remaining_epochs
            self.write_to_log(
                f'********** epoch_time: {self.format_eta(epoch_time)} | '
                f'total_eta: {self.format_eta(total_eta)} *********')

            # 保存权重
            if self.params['save_checkpoint']:
                # checkpoint at the end of every epoch
                self.save_checkpoint(self.params['checkpoint_path'] + 'ckpt_' +
                                     str(self.epoch) + '.tar')
                self.write_to_log_valid('save checkpoint to ' +
                                        self.params['checkpoint_path'])
                if valid_logs['valid_loss'] <= self.best_valid_loss:
                    self.write_to_log_valid(
                        'Val loss improved from {} to {}'.format(
                            self.best_valid_loss, valid_logs['valid_loss']))
                    self.save_checkpoint(self.params['best_checkpoint_path'])
                    self.best_valid_loss = valid_logs['valid_loss']

    def train_one_epoch(self):
        self.model.train()
        num_batches = len(self.dataloader)
        epoch_start = time.time()
        for i_batch, train_input_temp in enumerate(self.dataloader):
            """
            加载数据阶段
            """
            glory_batch, gt = train_input_temp

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
            gt[torch.where(torch.isnan(gt))] = 0
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

            _, prediction = self.model(bg_field)
            prediction = prediction.permute(0, 2, 3, 1)

            loss = self.loss(prediction, gt)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if i_batch % 10 == 0:
                logs = {'loss': loss}
                elapsed = time.time() - epoch_start
                done_batches = i_batch + 1
                avg_batch_time = elapsed / done_batches
                remaining_batches = num_batches - done_batches
                eta = avg_batch_time * remaining_batches
                log_message = (
                    f'[{self.epoch}/{self.max_epochs}] [{i_batch}/{num_batches}]  '
                    f'prediction: {loss} | eta: {self.format_eta(eta)}')
                self.write_to_log(log_message)

        return logs

    def validate_one_epoch(self):
        valid_loss_fine_list = []

        self.model.eval()

        for idx, (test_input_temp) in enumerate(self.dataloader_test):

            glory_batch, gt, _, _ = test_input_temp

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
            gt[torch.where(torch.isnan(gt))] = 0
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
                _, prediction = self.model(bg_field)
            prediction = prediction.permute(0, 2, 3, 1)

            loss = self.loss(prediction, gt)

            print('Validation: [{}/{}]  prediction: {}'.format(
                idx, len(self.dataloader_test), loss))

            valid_loss_fine_list.append(loss.cpu().detach().numpy())

            # save first channel of image
            if self.params['save_image'] and idx % 80 == 0:
                padding = 30
                for channel in range(gt.shape[3]):
                    # 创建保存目录

                    save_dir = self.params['exp_dir'] + '/' + self.params[
                        'run_num'] + "/vis_result/" + str(
                            self.epoch) + "/" + str(idx) + "_c" + str(
                                channel) + ".png"
                    if not os.path.exists(self.params['exp_dir'] + '/' +
                                          self.params['run_num'] +
                                          "/vis_result/" + str(self.epoch)):
                        os.makedirs(self.params['exp_dir'] + '/' +
                                    self.params['run_num'] + "/vis_result/" +
                                    str(self.epoch))

                    # 将输入和目标图像转换为 numpy 数组
                    input_image = bg_field[0, channel, :, :].cpu().numpy()
                    gen_image = prediction[0, :, :,
                                           channel].cpu().detach().numpy()
                    gt_image = gt[0, :, :, channel].cpu().numpy()

                    gen_image[gt_image == 0] = np.nan
                    input_image[gt_image == 0] = np.nan
                    gt_image[gt_image == 0] = np.nan

                    # 获取图像的高度和宽度
                    h, w = input_image.shape

                    # 创建空白区域（填充），这里使用nan数组
                    blank_space = np.full((h, padding), np.nan)

                    # 拼接：输入 | 生成 | 真实
                    combined_image = np.concatenate(
                        (input_image, blank_space, gen_image, blank_space,
                         gt_image),
                        axis=1)

                    # 创建图像
                    plt.figure(figsize=(80, 20))
                    plt.imshow(combined_image, cmap='bwr')

                    # 叠加传感器点（分别叠加到三幅图上）
                    x_sens = 0
                    y_sens = 0

                    # 左：input，无偏移
                    plt.scatter(y_sens, x_sens, color='blue', s=50)
                    # 中：gen，偏移一个面板宽度+padding
                    plt.scatter(y_sens + w + padding,
                                x_sens,
                                color='blue',
                                s=50)
                    # 右：gt，偏移两个面板宽度+2*padding
                    plt.scatter(y_sens + 2 * w + 2 * padding,
                                x_sens,
                                color='blue',
                                s=50)

                    plt.axis('off')

                    # 保存图像
                    plt.savefig(save_dir,
                                bbox_inches='tight',
                                pad_inches=0,
                                dpi=100)
                    plt.close()

                    break

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
                        default='./config/en4_ResNet_bg_model.yaml',
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
    params['best_checkpoint_path'] = os.path.join(
        expDir, 'training_checkpoints/best_ckpt.tar')

    trainer = Trainer(params)
    trainer.train()
