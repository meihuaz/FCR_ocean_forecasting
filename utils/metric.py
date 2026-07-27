import torch.nn.functional as F
import torch.nn as nn
import torch
import numpy as np

# class MAELoss(object):
#     def __init__(self):
#         super(MAELoss, self).__init__()
#         self.mae_loss = nn.L1Loss()  # 使用L1损失（MAE）

#     def __call__(self, x, y):
#         loss = self.mae_loss(x, y)
#         return loss
    

class MAELoss(object):
    def __init__(self):
        super(MAELoss, self).__init__()
        self.mae_loss = nn.L1Loss(reduction='none')  # 使用L1损失（MAE）

    def __call__(self, x, y):

        mask = ~(y==0)

        loss = self.mae_loss(x, y)
        loss = loss[mask].mean()


        return loss
    

    

class MAELoss_metric(object):
    def __init__(self):
        super(MAELoss_metric, self).__init__()
        self.mae_loss = nn.L1Loss(reduction='none')  # 使用L1损失（MAE）

    def __call__(self, x, y):

        y = y.cpu().numpy()

        mask = ~(np.isnan(y)) & ~(np.isnan(x.detach().cpu().numpy()))
        mask = torch.from_numpy(mask).to(x.device)
        y = torch.from_numpy(y).to(x.device)


        loss = self.mae_loss(x, y)
        loss = loss[mask].mean()


        return loss
    


class MAELoss_metric_test(object):
    def __init__(self):
        super(MAELoss_metric_test, self).__init__()
        self.mae_loss = nn.L1Loss(reduction='none')  # 使用L1损失（MAE）

    def __call__(self, x1, x2, y):

        y = y.cpu().numpy()

        mask = ~(np.isnan(y)) & ~(np.isnan(x1.detach().cpu().numpy()))
        mask = torch.from_numpy(mask).to(x1.device)
        y = torch.from_numpy(y).to(x1.device)


        loss1 = self.mae_loss(x1, y)
        loss1 = loss1[mask]
        

        loss2 = self.mae_loss(x2, y)
        loss2 = loss2[mask]


        loss1 = loss1.mean()
        loss2 = loss2.mean()


        return loss1
    
    

def RMSE_metric(y_pred, y_true):
    """
    计算均方根误差 (RMSE).

    参数:
    y_true (torch.Tensor): 真实值张量
    y_pred (torch.Tensor): 预测值张量

    返回:
    torch.Tensor: 计算得到的 RMSE 值
    """
    if type(y_pred) == np.ndarray:
        y_pred = torch.from_numpy(y_pred)
    if type(y_true) == np.ndarray:
        y_true = torch.from_numpy(y_true)

    # 计算均方误差 (MSE)
    mask = ~torch.isnan(y_true) & ~torch.isnan(y_pred)
    mse = torch.mean((y_pred[mask] - y_true[mask]) ** 2)

    # 计算均方根误差 (RMSE)
    rmse = torch.sqrt(mse)

    return rmse


def RMSE_metric(y_pred, y_true):
    """
    计算均方根误差 (RMSE).

    参数:
    y_true (torch.Tensor): 真实值张量
    y_pred (torch.Tensor): 预测值张量

    返回:
    torch.Tensor: 计算得到的 RMSE 值
    """
    if type(y_pred) == np.ndarray:
        y_pred = torch.from_numpy(y_pred)
    if type(y_true) == np.ndarray:
        y_true = torch.from_numpy(y_true)

    # 计算均方误差 (MSE)
    # mask = ~(y_true==0)
    mask = ~torch.isnan(y_true) & ~torch.isnan(y_pred)
    mse = torch.mean((y_pred[mask] - y_true[mask]) ** 2)

    # 计算均方根误差 (RMSE)
    rmse = torch.sqrt(mse)

    return rmse
