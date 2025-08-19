import sys, json
import torch
import torch.nn as nn
import numpy as np

sys.path.append('./Spec2vecModels/')
import params



def give_Loss_Function(loss_name, model_name, train_path=None, device=None):

    if   loss_name == "chi2":
        n_bins = params.n_bins_def if model_name not in params.n_bins_models.keys() else params.n_bins_models[model_name]
        return Chi2Loss(params.Csigma_chi2, n_bins)

    elif loss_name == "MSE": 
        return nn.MSELoss()

    elif loss_name == "L1N":
        return L1NLoss()

    elif loss_name == "MSEf3":
        if train_path is None : raise Exception("In [give_Loss_Function.py], train_path is needed for loss MSEf3 (here is None)")
        if device is None : raise Exception("In [give_Loss_Function.py], device is needed for loss MSEf3 (here is None)")
        with open(f"{train_path}/hist_params.json", 'r') as f:
            hp = json.load(f)
        ozone, pwv, aerosols = hp["ATM_OZONE"], hp["ATM_PWV"], hp["ATM_AEROSOLS"] 
        
        print(f"Load from {train_path}/hist_params.json :")
        print(f"ATM_OZONE    : {ozone}")
        print(f"ATM_PWV      : {pwv}")
        print(f"ATM_AEROSOLS : {aerosols}")

        return MSEf3Loss(ozone, pwv, aerosols, device)

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



class MSEf3Loss(nn.Module):

    def __init__(self, ozone, pwv, aerosols, device):
        super(MSEf3Loss, self).__init__()
        self.o = ozone
        self.p = pwv
        self.a = aerosols
        self.opaMIN = np.array([ozone[0], pwv[0], aerosols[0]])
        self.opaMAX = np.array([ozone[1], pwv[1], aerosols[1]])
        
        self.opaDEN = self.opaMAX - self.opaMIN
        self.opaDEN = torch.tensor(self.opaDEN, dtype=torch.float32).to(device)


    def forward(self, y_pred, y_true):

        diff = (y_pred - y_true)
        l1 = diff / self.opaDEN
        loss = torch.sum(torch.pow(l1, 2))

        # print(f"forward MSEf3Loss : {y_pred} \n {y_true} \n {diff} \n {l1} : {l1s} -2> {loss}")

        return loss