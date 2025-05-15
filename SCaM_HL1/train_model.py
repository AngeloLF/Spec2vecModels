from model import SCaM_Model, SCaM_Dataset
import params
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os, sys
import shutil
import numpy as np
from tqdm import tqdm
import coloralf as c

# Vérifier si CUDA est disponible et définir l'appareil
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Utilisation de l'appareil : {device}")


# find folders train & valid with sys argv
folder_train = None
folder_valid = None
num_epochs = params.num_epochs

for argv in sys.argv[1:]:

    if argv[:6] == "train=" : folder_train = argv[6:]
    if argv[:6] == "valid=" : folder_valid = argv[6:]
    if argv[:6] == "epoch=" : num_epochs = int(argv[6:])

# Define train folder
if folder_train is None:
    print(f"{c.r}WARNING : train folder is not define (train=<folder_train>){c.d}")
    raise ValueError("Train folder error")

# Define train folder
if folder_valid is None:
    print(f"{c.r}WARNING : valid folder is not define (train=<folder_valid>){c.d}")
    raise ValueError("Valid folder error")

# Hyperparamètres
batch_size = params.batch_size
learning_rate = params.lr_init
name = params.name

# Chemins des dossiers de données
path = params.path
train_simu_name = folder_train
valid_simu_name = folder_valid
fold_image, fold_spectrum = params.folder_image, params.folder_spectrum

train_image_dir    = f"{path}/{train_simu_name}/{fold_image}"
train_spectrum_dir = f"{path}/{train_simu_name}/{fold_spectrum}"
valid_image_dir    = f"{path}/{valid_simu_name}/{fold_image}"
valid_spectrum_dir = f"{path}/{valid_simu_name}/{fold_spectrum}"

# where to put all model training stuff
full_out_path = f"{params.out_path}/{params.out_dir}/{name}"
output_loss  = f"{full_out_path}/{params.out_loss}"
output_state = f"{full_out_path}/{params.out_states}"
output_epoch = f"{full_out_path}/{params.out_epoch}"

# if params.out_path not in os.listdir() : os.mkdir(params.out_path)
if params.out_dir not in os.listdir(params.out_path) : os.mkdir(f"{params.out_path}/{params.out_dir}")
if name not in os.listdir(f"{params.out_path}/{params.out_dir}") : os.mkdir(full_out_path)
if params.out_epoch in os.listdir(full_out_path) : shutil.rmtree(f"{full_out_path}/{params.out_epoch}")

for f in [params.out_loss, params.out_states, params.out_epoch]:
    if f not in os.listdir(f"{full_out_path}") : os.mkdir(f"{full_out_path}/{f}")



# Créer le Dataset complet
train_dataset = SCaM_Dataset(train_image_dir, train_spectrum_dir)
valid_dataset = SCaM_Dataset(valid_image_dir, valid_spectrum_dir)


# Créer les DataLoaders pour l'entraînement et la validation
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)


# Initialiser le modèle, la fonction de perte et l'optimiseur
model = SCaM_Model().to(device)
criterion = nn.HuberLoss(delta=0.1)  # On teste avec la HuberLoss
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

np.save(f"{output_loss}/{folder_train}.npy", np.array((train_list_loss, valid_list_loss)))
torch.save(best_state, f"{output_state}/{folder_train}.pth")   
