import torch
import torch.nn as nn



class Chi2Loss(nn.Module):

    def __init__(self, C, norma):
        super(Chi2Loss, self).__init__()
        self.C2 = C ** 2
        self.norma = norma

    def forward(self, y_pred, y_true):
        nume = torch.pow(y_true - y_pred, 2)
        deno = y_true + self.C2
        loss = torch.sum(nume / deno) / self.norma
        return loss