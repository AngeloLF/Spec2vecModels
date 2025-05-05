from model import SCaM_Model, SCaM_Dataset
import params
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os
import shutil
import numpy as np
from tqdm import tqdm
import coloralf as c

# Vérifier si CUDA est disponible et définir l'appareil
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Utilisation de l'appareil : {device}")

# Hyperparamètres
batch_size = params.batch_size
learning_rate = params.lr_init
num_epochs = params.num_epochs
name = params.name

# Chemins des dossiers de données
path = params.path
train_simu_name = params.folder_train
valid_simu_name = params.folder_valid
fold_image, fold_spectrum = params.folder_image, params.folder_spectrum

train_image_dir    = f"{path}/{train_simu_name}/{fold_image}"
train_spectrum_dir = f"{path}/{train_simu_name}/{fold_spectrum}"
valid_image_dir    = f"{path}/{valid_simu_name}/{fold_image}"
valid_spectrum_dir = f"{path}/{valid_simu_name}/{fold_spectrum}"

# where to put all model training stuff
output_loss = f"./{params.out_path}/{name}/{params.out_loss}"
output_state = f"./{params.out_path}/{name}/{params.out_states}"
output_epoch = f"./{params.out_path}/{name}/{params.out_epoch}"

# if params.out_path not in os.listdir() : os.mkdir(params.out_path)
if name not in os.listdir(params.out_path) : os.mkdir(f"{params.out_path}/{name}")
if params.out_epoch in os.listdir(f"{params.out_path}/{name}") : shutil.rmtree(f"{params.out_path}/{name}/{params.out_epoch}")

for f in [params.out_loss, params.out_states, params.out_epoch]:
    if f not in os.listdir(f"{params.out_path}/{name}") : os.mkdir(f"{params.out_path}/{name}/{f}")



# Créer le Dataset complet
train_dataset = SCaM_Dataset(train_image_dir, train_spectrum_dir)
valid_dataset = SCaM_Dataset(valid_image_dir, valid_spectrum_dir)


# Créer les DataLoaders pour l'entraînement et la validation
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)


# Initialiser le modèle, la fonction de perte et l'optimiseur
model = SCaM_Model().to(device)
criterion = nn.MSELoss()  # Mean Squared Error pour la régression du spectre
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

train_list_loss = np.zeros(num_epochs)
valid_list_loss = np.zeros(num_epochs)

best_val_loss = np.inf
best_state = None

print(
        "number of parameters is ",
        sum(p.numel() for p in model.parameters() if p.requires_grad) // 10**6,
        "millions",
    )

# Boucle d'entraînement
for epoch in range(num_epochs):

    # Boucle de training
    model.train()
    train_loss = 0.0
    for images, spectra in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} (Train)"):
        images = images.to(device)
        spectra = spectra.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, spectra)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * images.size(0)

    train_loss = train_loss / len(train_dataset)

    # Boucle d'évaluation sur l'ensemble de validation
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for images, spectra in tqdm(valid_loader, desc=f"Epoch {epoch+1}/{num_epochs} (Validation)"):
            images = images.to(device)
            spectra = spectra.to(device)
            outputs = model(images)
            loss = criterion(outputs, spectra)
            val_loss += loss.item() * images.size(0)

    val_loss = val_loss / len(valid_dataset)
    print(f"Epoch [{epoch+1}/{num_epochs}], loss train = {c.g}{train_loss:.6f}{c.d}, val loss = {c.r}{val_loss:.6f}{c.d}")
    with open(f"{output_epoch}/INFO - epoch {epoch+1} - {train_loss:.6f} , {val_loss:.6f}", "wb") as f : pass

    # save state at each epoch to be able to reload and continue the optimization
    if val_loss < best_val_loss:

        best_val_loss = val_loss

        best_state = {
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        }

    train_list_loss[epoch] = train_loss
    valid_list_loss[epoch] = val_loss

print("Entraînement terminé.")

np.save(f"{output_loss}/{params.folder_train}.npy", np.array((train_list_loss, valid_list_loss)))
torch.save(best_state, f"{output_state}/{params.folder_train}.pth")   
