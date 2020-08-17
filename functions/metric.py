import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import multilabel_confusion_matrix

from .loss import OneHotEncoder


class DiceCoefficient(object):
    epsilon = 1e-5

    def __init__(self, n_classes, index_to_class_name, ignore_index=None):
        super().__init__()
        self.one_hot_encoder = OneHotEncoder(n_classes).forward
        self.n_classes = n_classes
        self.ignore_index = ignore_index
        self.index_to_class_name = index_to_class_name

    def update(self, output, target):
        batch_size = output.shape[0]

        output = self.one_hot_encoder(torch.argmax(output, dim=1))
        output = output.contiguous().view(batch_size, self.n_classes, -1)
        target = target.view(batch_size, self.n_classes, -1)
        assert output.shape == target.shape

        dice = {}
        for i in range(self.n_classes):
            if i == self.ignore_index:
                continue

            os = output[:, i, ...]
            ts = target[:, i, ...]

            inter = torch.sum(os * ts, dim=1)
            union = torch.sum(os, dim=1) + torch.sum(ts, dim=1)
            score = torch.sum(2.0 * inter / union.clamp(min=self.epsilon))
            score /= batch_size

            if self.index_to_class_name:
                dice[self.index_to_class_name[i]] = score.item()
            else:
                dice[str(i)] = score.item()

        return dice 
