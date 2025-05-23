import os, sys, shutil, importlib
import numpy as np
from tqdm import tqdm
import coloralf as c

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

sys.path.append('./Spec2vecModels/')
import params
from losses import give_Loss_Function, Chi2Loss




if __name__ == "__main__":


    ### capture params
    model_name = None
    folder_train = None
    folder_valid = None
    loss_name = None
    num_epochs = params.num_epochs
    learning_rate = params.lr_default

    for argv in sys.argv[1:]:

        if argv[:6] == "model=" : model_name = argv[6:]
        if argv[:6] == "train=" : folder_train = argv[6:]
        if argv[:6] == "valid=" : folder_valid = argv[6:]
        if argv[:6] == "epoch=" : num_epochs = int(argv[6:])
        if argv[:3] == "lr="    : learning_rate = float(argv[3:])
        if argv[:5] == "loss="  : loss_name = argv[5:]



    ### Import model & dataset
    if model_name is None:
        print(f"{c.r}WARNING : model name is not define (model=<model_name>){c.d}")
        raise ValueError("Model name error")
    else:
        module_name = f"{model_name}"

        print(f"{c.y}INFO : Import module {module_name} ...{c.d}")
        module = importlib.import_module(f"architecture.{module_name}")

        print(f"{c.y}INFO : Import model {model_name}_Model et le dataloader {model_name}_Dataset ...{c.d}")
        model = getattr(module, f"{model_name}_Model")
        perso_dataloader = getattr(module, f"{model_name}_Dataset")



    ### Define train folder
    if folder_train is None:
        print(f"{c.r}WARNING : train folder is not define (train=<folder_train>){c.d}")
        raise ValueError("Train folder error")


    ### Define valid folder
    if folder_valid is None:
        print(f"{c.r}WARNING : valid folder is not define (valid=<folder_valid>){c.d}")
        raise ValueError("Valid folder error")



    ### Define loss function
    if loss_name is None:
        print(f"{c.r}WARNING : loss function is not define (loss=<loss_name>){c.d}")
        raise ValueError("Loss function error")
    else:
        loss_function = give_Loss_Function(loss_name)



    ### Define some params
    name = f"{model_name}_{loss_name}"
    batch_size = params.batch_size



    ### Definition of paths
    path = params.path
    train_name = f"{folder_train}_{learning_rate:.0e}"
    full_out_path     = f"{params.out_path}/{params.out_dir}/{name}"

    train_inp_dir = f"{path}/{folder_train}/{model.folder_input}"
    train_out_dir = f"{path}/{folder_train}/{model.folder_output}"
    valid_inp_dir = f"{path}/{folder_valid}/{model.folder_input}"
    valid_out_dir = f"{path}/{folder_valid}/{model.folder_output}"

    # Define path of losses
    output_loss       = f"{full_out_path}/{params.out_loss}"
    output_loss_mse   = f"{full_out_path}/{params.out_loss_mse}"
    output_loss_chi2  = f"{full_out_path}/{params.out_loss_chi2}"
    output_state      = f"{full_out_path}/{params.out_states}"
    output_divers     = f"{full_out_path}/{params.out_divers}"

    # Create folder in case ...
    os.makedirs(f"{params.out_path}/{params.out_dir}", exist_ok=True)
    os.makedirs(full_out_path, exist_ok=True)
    for f in [params.out_loss, params.out_loss_mse, params.out_loss_chi2, params.out_states, params.out_divers]:
        os.makedirs(f"{full_out_path}/{f}", exist_ok=True)

    # Folder for push epoch
    output_epoch      = f"{full_out_path}/{params.out_epoch}"
    output_epoch_here = f"{output_epoch}/{train_name}"
    os.makedirs(output_epoch, exist_ok=True)
    if train_name in os.listdir(output_epoch) : shutil.rmtree(output_epoch_here)
    os.mkdir(output_epoch_here)



    ### Data set loading

    # Créer le Dataset complet
    train_dataset = perso_dataloader(train_inp_dir, train_out_dir)
    valid_dataset = perso_dataloader(valid_inp_dir, valid_out_dir)

    # Créer les DataLoaders pour l'entraînement et la validation
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)



    ### Initialiser le modèle, la fonction de perte et l'optimiseur
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model().to(device)
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
            loss = loss_function(outputs, spectra)
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
                loss = loss_function(outputs, spectra)
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

    if "nosave" not in sys.argv[1:] : torch.save(best_state, f"{output_state}/{folder_train}_{learning_rate:.0e}_best.pth") 

