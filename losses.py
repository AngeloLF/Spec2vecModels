import sys
import torch
import torch.nn as nn

sys.path.append('./Spec2vecModels/')
import params



def give_Loss_Function(loss_name, model_name):

    if   loss_name == "chi2":
        n_bins = params.n_bins_def if model_name not in params.n_bins_models.keys() else params.n_bins_models[model_name]
        return Chi2Loss(params.Csigma_chi2, n_bins)

    elif loss_name == "MSE": 
        return nn.MSELoss()

    elif loss_name == "L1N":
        return L1NLoss()
        
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



class L1NLoss(nn.Module):

    def __init__(self):
        super(L1NLoss, self).__init__()

    def forward(self, y_pred, y_true):
        return torch.sum(torch.abs(y_true-y_pred)) / torch.sum(y_true)