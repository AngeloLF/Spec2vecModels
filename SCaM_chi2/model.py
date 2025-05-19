import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
from tqdm import tqdm



# Classe pour le Dataset personnalisé
class SCaM_Dataset(Dataset):
    def __init__(self, image_dir, spectrum_dir):
        self.image_files = sorted([os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith(".npy")])
        self.spectrum_files = sorted([os.path.join(spectrum_dir, f) for f in os.listdir(spectrum_dir) if f.endswith(".npy")])

        # prof si le nombre est différent ....
        assert len(self.image_files) == len(self.spectrum_files), "Le nombre de fichiers images / spectrums ne correspond pas"

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image = np.load(self.image_files[idx]).astype(np.float32)
        spectrum = np.load(self.spectrum_files[idx]).astype(np.float32)
        # add une dimension de canal pour le CNN (1 canal car image en niveaux de gris implicite)
        image = np.expand_dims(image, axis=0)
        return torch.tensor(image), torch.tensor(spectrum)


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


class SCaM_Model(nn.Module):
    """
    SCaM Model : Simple CNN and MLP
    """

    nameOfThisModel = "SCaM_chi2"
    
    def __init__(self):

        super(SCaM_Model, self).__init__()
        # Image d'entrée : 1x128x1024
        
        # 1ere convolution CNN
        self.conv1 = nn.Conv2d(1, 16, kernel_size=(3, 3), padding=(1, 1))  # -> 16x128x1024
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))       # -> 16x64x512

        # 2eme convolution CNN
        self.conv2 = nn.Conv2d(16, 32, kernel_size=(3, 3), padding=(1, 1)) # -> 32x64x512
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))       # -> 32x32x256

        # 3eme convolution CNN
        self.conv3 = nn.Conv2d(32, 64, kernel_size=(3, 3), padding=(1, 1)) # -> 64x32x256
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))       # -> 64x16x128

        # On applatit les dernier filtre
        self.flatten = nn.Flatten() # -> 131072

        # MLP
        self.fc1 = nn.Linear(64 * 16 * 128, 1024) # 131072 -> 1024
        self.relu4 = nn.ReLU()
        self.fc2 = nn.Linear(1024, 800)           # 131072 -> 800

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.pool3(self.relu3(self.conv3(x)))
        x = self.flatten(x)
        x = self.relu4(self.fc1(x))
        x = self.fc2(x)
        return x