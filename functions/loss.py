import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np


class SoftDiceLoss(nn.Module):

    def __init__(self, ignore_index=None, smooth=1.0, reduction='mean'):
        super().__init__()
        self.ignore_index = ignore_index
        self.smooth = smooth
        self.reduction = reduction

    def forward(self, output, target):
        # output: (B, C, H, W, D)
        # target: (B, C, H, W, D)
        assert output.shape == target.shape
        batch_size = output.shape[0]

        loss = 0.0
        n_count = 0
        for i in range(output.shape[1]):
            if self.ignore_index is not None and i == self.ignore_index:
                continue

            os = output[:, i, ...].clone()
            os = os.view(batch_size, -1)
            ts = target[:, i, ...].clone()
            ts = ts.view(batch_size, -1).float()

            inter = (os * ts).sum()
            union = os.sum() + ts.sum()

            loss += 1 - (2 * inter + self.smooth) / (union + self.smooth)
            n_count += 1

        if self.reduction == 'mean':
            loss /= n_count

        return loss


class FocalLoss(nn.Module):
    epsilon = 1e-8

    def __init__(self, gamma=2, alpha=None, ignore_index=None, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.ignore_index = ignore_index
        self.reduction = reduction

        self.criterion = nn.BCELoss(reduction='none')

    def forward(self, output, target):
        # output: (B, C, H, W, D)
        # target: (B, C, H, W, D)
        assert output.shape == target.shape
        batch_size = output.shape[0]

        loss = 0.0
        n_count = 0
        for i in range(output.shape[1]):
            if self.ignore_index is not None and i == self.ignore_index:
                continue

            os = output[:, i, ...].clone()
            os = os.view(batch_size, -1).clamp(min=self.epsilon, max=1 - self.epsilon)
            ts = target[:, i, ...].clone()
            ts = ts.view(batch_size, -1).float()

            logpt_pos = ts * torch.log(os)
            pt_pos = torch.exp(logpt_pos)

            if self.alpha:
                logpt_pos *= self.alpha

            val = - ((1 - pt_pos) ** self.gamma) * logpt_pos

            # logpt_neg = (1 - ts) * torch.log(1 - os)
            # pt_neg = torch.exp(logpt_neg)
            #
            # if self.alpha:
            #     logpt_neg *= self.alpha
            #
            # val += - ((1 - pt_neg) ** self.gamma) * logpt_neg

            loss += val.mean()
            n_count += 1

        if self.reduction == 'mean':
            loss /= n_count

        return loss


class ActiveContourLoss(nn.Module):

    def __init__(self, weight=1, epsilon=1e-8, ignore_index=None, reduction='mean'):
        super().__init__()
        self.weight = weight
        self.epsilon = epsilon
        self.ignore_index = ignore_index
        self.reduction = reduction

    def forward(self, output, target):
        # output: (B, C, H, W, D)
        # target: (B, C, H, W, D)
        assert output.shape == target.shape
        batch_size = output.shape[0]

        loss = 0.0
        n_count = 0
        for i in range(output.shape[1]):
            if self.ignore_index is not None and i == self.ignore_index:
                continue

            os = output[:, i, ...].clone()
            ts = target[:, i, ...].clone()

            os[os >= 0.5] = 1
            os[os < 0.5] = 0

            # length term
            delta_r = os[:, 1:, :] - os[:, :-1, :]  # horizontal gradient (B, H-1, W)
            delta_c = os[:, :, 1:] - os[:, :, :-1]  # vertical gradient (B, H, W-1)

            delta_r = delta_r[:, 1:, :-2] ** 2  # (B, H-2, W-2)
            delta_c = delta_c[:, :-2, 1:] ** 2  # (B, H-2, W-2)

            delta_pred = torch.abs(delta_r + delta_c)
            length = torch.mean(torch.sqrt(delta_pred + self.epsilon))

            # region term
            c_in = torch.ones_like(os)
            c_out = torch.zeros_like(os)

            region_in = torch.mean(os * (ts - c_in) ** 2)
            region_out = torch.mean((1 - os) * (ts - c_out) ** 2)
            region = region_in + region_out

            loss += self.weight * length + region
            n_count += 1

        if self.reduction == 'mean':
            loss /= n_count

        return loss


class OneHotEncoder(nn.Module):

    def __init__(self, n_classes):
        super().__init__()

        self.n_classes = n_classes
        self.ones = torch.sparse.torch.eye(n_classes).cuda()

    def forward(self, t):
        n_dim = t.dim()
        output_size = t.size() + torch.Size([self.n_classes])

        t = t.data.long().contiguous().view(-1).cuda()
        out = Variable(self.ones.index_select(0, t)).view(output_size)
        out = out.permute(0, -1, *range(1, n_dim)).float()

        return out
