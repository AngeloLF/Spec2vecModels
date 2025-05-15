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

        # Assurez-vous que le nombre de fichiers images et spectres correspond
        assert len(self.image_files) == len(self.spectrum_files), "Le nombre de fichiers images et spectres ne correspond pas."

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image = np.load(self.image_files[idx]).astype(np.float32)
        spectrum = np.load(self.spectrum_files[idx]).astype(np.float32)
        # Ajouter une dimension de canal pour le CNN (1 canal car image en niveaux de gris implicite)
        image = np.expand_dims(image, axis=0)
        return torch.tensor(image), torch.tensor(spectrum)

# Définition du modèle CNN + MLP
class SCaM_Model(nn.Module):
    
    def __init__(self):

        self.nameOfThisModel = "SCaM"

        super(SCaM_Model, self).__init__()
        # CNN pour extraire les caractéristiques de l'image 1x128x1024
        self.conv1 = nn.Conv2d(1, 16, kernel_size=(3, 3), padding=(1, 1))
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        self.conv2 = nn.Conv2d(16, 32, kernel_size=(3, 3), padding=(1, 1))
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        self.conv3 = nn.Conv2d(32, 64, kernel_size=(3, 3), padding=(1, 1))
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        # Calculer la taille de la sortie après les convolutions et le pooling
        # La taille après pool1 sera 16 x 64 x 512
        # La taille après pool2 sera 32 x 32 x 256
        # La taille après pool3 sera 64 x 16 x 128
        self.flatten = nn.Flatten()

        # MLP pour mapper les caractéristiques extraites au spectre de sortie de taille 800
        # La taille de l'entrée du MLP dépend de la sortie aplatie du CNN
        self.fc1 = nn.Linear(64 * 16 * 128, 1024)
        self.relu4 = nn.ReLU()
        self.fc2 = nn.Linear(1024, 800)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.pool3(self.relu3(self.conv3(x)))
        x = self.flatten(x)
        x = self.relu4(self.fc1(x))
        x = self.fc2(x)
        return x


# Vous pouvez maintenant utiliser le modèle entraîné pour prédire les spectres à partir de nouvelles images de test.
# Exemple de prédiction pour une image de test :
# model.eval()
# with torch.no_grad():
#     test_image, _ = test_dataset[0]
#     test_image = test_image.unsqueeze(0).to(device) # Ajouter la dimension du batch
#     predicted_spectrum = model(test_image).cpu().numpy()
#     print("Spectre prédit:", predicted_spectrum.shape)