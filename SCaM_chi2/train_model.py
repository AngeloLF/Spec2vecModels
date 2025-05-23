import os, sys, shutil
import numpy as np
from tqdm import tqdm
import coloralf as c

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

sys.path.append('./Spec2vecModels/')
import params
from losses import Chi2Loss
from architecture_SCaM import SCaM_Model, SCaM_Dataset as Current_Model, Current_Dataset




# Parameters of THIS model
name = "SCaM_chi2"
func_loss = Chi2Loss(params.Csigma_chi2, params.n_bins)
batch_size = params.batch_size




### find folders train & valid with sys argv

folder_train = None
folder_valid = None
num_epochs = params.num_epochs
learning_rate = params.lr_default

for argv in sys.argv[1:]:

    if argv[:6] == "train=" : folder_train = argv[6:]
    if argv[:6] == "valid=" : folder_valid = argv[6:]
    if argv[:6] == "epoch=" : num_epochs = int(argv[6:])
    if argv[:3] == "lr="    : learning_rate = float(argv[3:])

# Define train folder
if folder_train is None:
    print(f"{c.r}WARNING : train folder is not define (train=<folder_train>){c.d}")
    raise ValueError("Train folder error")

# Define train folder
if folder_valid is None:
    print(f"{c.r}WARNING : valid folder is not define (train=<folder_valid>){c.d}")
    raise ValueError("Valid folder error")




### Definition of paths
path = params.path
train_simu_name = folder_train
valid_simu_name = folder_valid
fold_image, fold_spectrum = params.folder_image, params.folder_spectrum
train_name = f"{folder_train}_{learning_rate:.0e}"

train_image_dir    = f"{path}/{train_simu_name}/{fold_image}"
train_spectrum_dir = f"{path}/{train_simu_name}/{fold_spectrum}"
valid_image_dir    = f"{path}/{valid_simu_name}/{fold_image}"
valid_spectrum_dir = f"{path}/{valid_simu_name}/{fold_spectrum}"

# where to put all model training stuff
full_out_path = f"{params.out_path}/{params.out_dir}/{name}"
output_loss  = f"{full_out_path}/{params.out_loss}"
output_loss_mse = f"{full_out_path}/{params.out_loss_mse}"
output_loss_chi2 = f"{full_out_path}/{params.out_loss_chi2}"
output_state = f"{full_out_path}/{params.out_states}"
output_divers = f"{full_out_path}/{params.out_divers}"
output_epoch = f"{full_out_path}/{params.out_epoch}"
output_epoch_here = f"{output_epoch}/{train_name}"

# if params.out_path not in os.listdir() : os.mkdir(params.out_path)
if params.out_dir not in os.listdir(params.out_path) : os.mkdir(f"{params.out_path}/{params.out_dir}")
if name not in os.listdir(f"{params.out_path}/{params.out_dir}") : os.mkdir(full_out_path)
if params.out_epoch not in os.listdir(full_out_path) : os.mkdir(output_epoch)
if train_name in os.listdir(output_epoch) : shutil.rmtree(output_epoch_here)
os.mkdir(output_epoch_here)

for f in [params.out_loss, params.out_loss_mse, params.out_loss_chi2, params.out_states, params.out_divers]:
    if f not in os.listdir(f"{full_out_path}") : os.mkdir(f"{full_out_path}/{f}")




### Data set loading

# Créer le Dataset complet
train_dataset = Current_Dataset(train_image_dir, train_spectrum_dir)
valid_dataset = Current_Dataset(valid_image_dir, valid_spectrum_dir)

# Créer les DataLoaders pour l'entraînement et la validation
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)


### Initialiser le modèle, la fonction de perte et l'optimiseur
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Current_Model().to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)




### Define losses

# Train loss
train_list_loss = np.zeros(num_epochs)
valid_list_loss = np.zeros(num_epochs)

# MSE loss
mse_loss = nn.MSELoss()
train_list_loss_mse = np.zeros(num_epochs)
valid_list_loss_mse = np.zeros(num_epochs)

# Chi2 loss
chi2_loss = Chi2Loss(params.Csigma_chi2, params.n_bins)
train_list_loss_chi2 = np.zeros(num_epochs)
valid_list_loss_chi2 = np.zeros(num_epochs)

# Save lr
lrates = np.zeros(num_epochs)

best_val_loss = np.inf
best_state = None




### Some print
print(f"{c.ly}INFO : Utilisation de l'appareil : {device}{c.d}")
print(f"{c.ly}INFO : Model architecture {name}{c.d}")
print(f"{c.ly}INFO : Name : {train_name}{c.d}")
print(f"{c.ly}INFO : Train : {folder_train}{c.d}")
print(f"{c.ly}INFO : Valid : {folder_valid}{c.d}")
print(f"{c.ly}INFO : Epoch : {num_epochs}{c.d}")
print(f"{c.ly}INFO : Lrate : {learning_rate}{c.d}")
print(f"{c.ly}INFO : Number of parameters : {sum(p.numel() for p in model.parameters() if p.requires_grad) // 10**6} millions{c.d}")




### Boucle d'entraînement

for epoch in range(num_epochs):



    ### Training 

    train_loss = 0.0
    train_loss_mse = 0.0
    train_loss_chi2 = 0.0

    for images, spectra in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} (Train)"):

        # Make the training
        model.train()
        images = images.to(device)
        spectra = spectra.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = func_loss(outputs, spectra)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * images.size(0)

        # Evaluate with mse and chi2 loss
        model.eval()
        train_loss_mse += mse_loss(outputs, spectra) * images.size(0)
        train_loss_chi2 += chi2_loss(outputs, spectra) * images.size(0)


    train_list_loss[epoch] = train_loss / len(train_dataset)
    train_list_loss_mse[epoch] = train_loss_mse / len(train_dataset)
    train_list_loss_chi2[epoch] = train_loss_chi2 / len(train_dataset)



    ### Validation

    model.eval()
    valid_loss = 0.0
    valid_loss_mse = 0.0
    valid_loss_chi2 = 0.0
    
    with torch.no_grad():

        for images, spectra in tqdm(valid_loader, desc=f"Epoch {epoch+1}/{num_epochs} (Validation)"):

            # classique validation
            images = images.to(device)
            spectra = spectra.to(device)
            outputs = model(images)
            loss = func_loss(outputs, spectra)
            valid_loss += loss.item() * images.size(0)

            # Evaluate with mse and chi2 loss
            model.eval()
            valid_loss_mse += mse_loss(outputs, spectra) * images.size(0)
            valid_loss_chi2 += chi2_loss(outputs, spectra) * images.size(0)

    valid_loss = valid_loss / len(valid_dataset)

    # Show epoch
    lrates[epoch] = optimizer.state_dict()['param_groups'][0]['lr']
    print(f"Epoch [{epoch+1}/{num_epochs}], loss train = {c.g}{train_loss:.6f}{c.d}, val loss = {c.r}{valid_loss:.6f}{c.d} | LR={c.y}{lrates[epoch]:.2e}{c.d}")
    with open(f"{output_epoch_here}/INFO - epoch {epoch+1} - {train_loss:.6f} , {valid_loss:.6f}", "wb") as f : pass

    # save state at each epoch to be able to reload and continue the optimization
    if valid_loss < best_val_loss:

        best_val_loss = valid_loss
        best_state = {"epoch": epoch + 1, "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict()}

    # save loss
    valid_list_loss[epoch] = valid_loss
    valid_list_loss_mse[epoch] = valid_loss_mse / len(valid_dataset)
    valid_list_loss_chi2[epoch] = valid_loss_chi2 / len(valid_dataset)


### save everything
print("Saving loss history and states of best models")

np.save(f"{output_loss}/{folder_train}_{learning_rate:.0e}.npy", np.array((train_list_loss, valid_list_loss)))
np.save(f"{output_loss_mse}/{folder_train}_{learning_rate:.0e}.npy", np.array((train_list_loss_mse, valid_list_loss_mse)))
np.save(f"{output_loss_chi2}/{folder_train}_{learning_rate:.0e}.npy", np.array((train_list_loss_chi2, valid_list_loss_chi2)))
np.save(f"{output_divers}/lr_{folder_train}_{learning_rate:.0e}.npy", lrates)

torch.save(best_state, f"{output_state}/{folder_train}_{learning_rate:.0e}_best.pth") 
