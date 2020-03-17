import numpy as np
import torch
import torch.nn.functional as F


class SiameseMetric(torch.nn.Module):
    """
    DeepFake competition metric: log-loss
    """

    def __init__(self):
        super(SiameseMetric, self).__init__()
        self.fn = F.cross_entropy

    def forward(self, outputs: torch.Tensor, labels: torch.Tensor, **kwargs):
        return self.fn(outputs, labels)


class DeepFakeMetric(torch.nn.Module):
    """
    DeepFake competition metric: log-loss
    """

    def __init__(self):
        super(DeepFakeMetric, self).__init__()
        self.fn = torch.nn.BCEWithLogitsLoss()
        # self.fn = F.cross_entropy

    def forward(self, outputs: torch.Tensor, labels: torch.Tensor, **kwargs):
        # outputs = torch.max(F.softmax(outputs, dim=1), dim=1)[0]
        return self.fn(outputs, labels)
        # outputs = torch.sigmoid(outputs).long()
        # labels = labels.long()
        # true = torch.mul(labels, torch.log(outputs)).sum()
        # false = torch.mul(torch.sub(1., labels), torch.log(torch.sub(1., outputs))).sum()
        # total = labels.size(0)
        # # correct = (labels == outputs).sum().float() / total
        # # return 100. * correct
        # acc = - (1 / total) * (true + false)
        # return acc.float()


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience"""

    def __init__(self, patience=7, verbose=False, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.save_checkpoint = False

    def __call__(self, val_loss, model=None):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint = True
            self.val_loss_min = val_loss
        elif score < self.best_score + self.delta:
            self.counter += 1
            # print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
            self.save_checkpoint = False
        else:
            self.best_score = score
            self.save_checkpoint = True
            self.val_loss_min = val_loss
            self.counter = 0
