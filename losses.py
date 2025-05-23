import sys
import torch
import torch.nn as nn

sys.path.append('./Spec2vecModels/')
import params



def give_Loss_Function(loss_name):

    if   loss_name.lower() == "chi2" : return Chi2Loss(params.Csigma_chi2, params.n_bins)
    elif loss_name.lower() == "mse" : return nn.MSELoss()
    else: 
        print(f"{c.r}WARNING : loss name {loss_name} unknow{c.d}")
        raise ValueError("Loss name unknow")



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