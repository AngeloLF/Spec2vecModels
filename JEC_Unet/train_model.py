import os, sys, json, shutil
import random
import pathlib
import pickle
from types import SimpleNamespace
import multiprocessing
import time
import params as args

import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import numpy as np
import regex as re
import glob

import coloralf as c
from model import UNet, CustumDataset


################
# Utils
################
def set_seed(seed):
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


################
# dataloader
################




################
# train/test 1-epoch
################
def train(args, model, criterion, train_loader, optimizer, epoch):
    model.train()
    loss_sum = 0  # to get the mean loss over the dataset
    for i_batch, sample_batched in enumerate(train_loader):
        # get the "X,y"
        imgs = sample_batched[0].to(args.device)
        spectra = sample_batched[1].to(args.device)

        # train step
        optimizer.zero_grad()
        output = model(imgs)
        loss = criterion(output, spectra)
        loss_sum += loss.item()
        # backprop to compute the gradients
        loss.backward()
        # perform an optimizer step to modify the weights
        optimizer.step()

    if args.archi == "Unet-Encoder":
        tmp2 = model.fc0.linear.weight.grad
        print("epoch",epoch,"ib",i_batch,
              "max_abs e.0.0 grad",torch.max(torch.abs( model.encoder["0"][0].weight.grad)),"\n",
              "max_abs e.0.2 grad",torch.max(torch.abs( model.encoder["0"][2].weight.grad)),"\n",
              "max_abs e.1.0 grad",torch.max(torch.abs( model.encoder["1"][0].weight.grad)),"\n",
              "max_abs e.1.3 grad",torch.max(torch.abs( model.encoder["1"][3].weight.grad)),"\n",
              "max_abs e.2.0 grad",torch.max(torch.abs( model.encoder["2"][0].weight.grad)),"\n",
              "max_abs e.2.3 grad",torch.max(torch.abs( model.encoder["2"][3].weight.grad)),"\n",
              "max_abs e.3.0 grad",torch.max(torch.abs( model.encoder["3"][0].weight.grad)),"\n",
              "max_abs e.3.3 grad",torch.max(torch.abs( model.encoder["3"][3].weight.grad)),"\n",
              "max_abs e.4.0 grad",torch.max(torch.abs( model.encoder["4"][0].weight.grad)),"\n",
              "max_abs e.4.3 grad",torch.max(torch.abs( model.encoder["4"][3].weight.grad)),"\n",
              "max_abs fc0 grad",torch.max(torch.abs(tmp2)),
              )
    elif args.archi == "Resnet18":
        print("epoch",epoch,"ib",i_batch,
              "max_abs l1.0.conv1 grad",torch.max(torch.abs(model.layer1[0].conv1.weight.grad)),"\n",
              "max_abs l4.1.conv2 grad",torch.max(torch.abs(model.layer4[1].conv2.weight.grad)),"\n",
              "max_abs fc grad",torch.max(torch.abs(model.fc.weight.grad)),
              )
        
        

    return loss_sum / (i_batch + 1)


def test(args, model, criterion, test_loader, epoch):
    model.eval()
    loss_sum = 0  # to get the mean loss over the dataset
    with torch.no_grad():
        for i_batch, sample_batched in enumerate(test_loader):
            # get the "X,y"
            imgs = sample_batched[0].to(args.device)
            spectra = sample_batched[1].to(args.device)
            #
            output = model(imgs)
            loss = criterion(output, spectra)
            loss_sum += loss.item()

    return loss_sum / (i_batch + 1)



if __name__ == "__main__":

    # check number of num_workers
    NUM_CORES = multiprocessing.cpu_count()

    if args.num_workers >= NUM_CORES:
        print("Info: # workers set to", NUM_CORES // 2)
        args.num_workers = NUM_CORES // 2

    name = args.name

    # where to find the dataset (train & test) (input images & output spectra)
    args.train_img_dir  = f"{args.data_path}/{args.folder_train}/{args.folder_images}"
    args.train_spec_dir = f"{args.data_path}/{args.folder_train}/{args.folder_spectrum}"
    args.test_img_dir   = f"{args.data_path}/{args.folder_test}/{args.folder_images}"
    args.test_spec_dir  = f"{args.data_path}/{args.folder_test}/{args.folder_spectrum}"

    # where to put all model training stuff
    output_loss = f"./{args.out_path}/{name}/{args.out_loss}"
    output_state = f"./{args.out_path}/{name}/{args.out_states}"
    output_epoch = f"./{args.out_path}/{name}/{args.out_epoch}"

    # if args.out_path not in os.listdir() : os.mkdir(args.out_path)
    if name not in os.listdir(args.out_path) : os.mkdir(f"{args.out_path}/{name}")
    if args.out_epoch in os.listdir(args.out_path) : shutil.rmtree(f"{args.out_path}/{args.out_epoch}")

    for f in [args.out_loss, args.out_states, args.out_epoch]:
        if f not in os.listdir(f"{args.out_path}/{name}") : os.mkdir(f"{args.out_path}/{name}/{f}")


    # device cpu/gpu...
    args.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"Device : {args.device}")

    # seeding
    set_seed(args.seed)

    # dataset & dataloader
    ds_train = CustumDataset(args.train_img_dir, args.train_spec_dir)
    ds_test = CustumDataset(args.test_img_dir, args.test_spec_dir)

    train_loader = DataLoader(
        dataset=ds_train,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=args.num_workers,
        drop_last=True,
    )
    test_loader = DataLoader(
        dataset=ds_test,
        batch_size=args.test_batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=args.num_workers,
        drop_last=True,
    )

    # get a batch to determin the image/spectrum sizes
    train_img, train_spec = next(iter(train_loader))
    #img_channels = args.num_channels
    img_H = train_img.shape[2]
    img_W = train_img.shape[3]
    args.n_bins = train_spec.shape[1]
    print("image sizes: HxW", img_H, img_W, "spectrum # bins", args.n_bins)

    # model instantiation
    model = UNet()

    # check ouptut of model is ok. Allow to determine the model config done at run tile
    out = model(train_img)
    assert out.shape == train_spec.shape
    print(
        "number of parameters is ",
        sum(p.numel() for p in model.parameters() if p.requires_grad) // 10**6,
        "millions",
    )
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name)

    # put model to device before loading scheduler/optimizer parameters
    model.to(args.device)

    # optimizer & scheduler == JEC 7 april 25
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr_init,
        #eps=1e-8, # by default is 1e-8
        #weight_decay=1e-3   # default is 0
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=args.lr_decay,
        patience=args.patience,
        min_lr=1e-6
    )

    # check for resume session: load model/optim/scheduler dictionnaries
    train_loss_history = []
    test_loss_history = []

    # loss
    criterion = nn.MSELoss(reduction="mean")

    # loop on epochs
    t0 = time.time()
    best_test_loss = np.inf
    best_state = None
    
    for epoch in range(args.num_epochs):

        # training
        train_loss = train(args, model, criterion, train_loader, optimizer, epoch)
        
        # test
        test_loss = test(args, model, criterion, test_loader, epoch)

        # print & book keeping
        print(f"Epoch {epoch+1} : Loss train = {c.g}{train_loss:.6f}{c.d}, loss test {c.r}{test_loss:.6f}{c.d}, LR={scheduler.get_last_lr()}")
        with open(f"{output_epoch}/INFO - epoch {epoch+1} - {train_loss:.6f} , {test_loss:.6f}", "wb") as f : pass
        train_loss_history.append(train_loss)
        test_loss_history.append(test_loss)

        # update scheduler
        if args.use_scheduler:
            # Warning ReduceLROnPlateau needs a metric
            scheduler.step(test_loss)

        # save state at each epoch to be able to reload and continue the optimization
        if test_loss < best_test_loss:

            best_test_loss = test_loss

            if args.use_scheduler:
                best_state = {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                }
            else:
                best_state = {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                }

    np.save(f"{output_loss}/{args.folder_train}.npy", np.array((train_loss_history, test_loss_history)))
    torch.save(best_state, f"{output_state}/{args.folder_train}.pth")   

    # Bye
    tf = time.time()
    print("all done!", tf - t0)
