# define JEC Unet params

def __getattr__(name):

    if name in locals() : return locals()[name]
    else : raise ValueError(f"params (of JEC_Unet) dont have {name=}.")


archi= "Unet-Encoder"
pool_window= 2
num_blocks= 5
num_kernels= 16
num_channels= 1
kernel_size= 3
padding= 1
bias= True
num_enc_conv= 2

data_path= "./../output_simu"
folder_train= "train256"
folder_test = "valid64"
folder_images= "image"
folder_spectrum = "spectrum"

out_path = "."
out_loss = "loss"
out_states = "states"
out_epoch = "epoch"

seed= 42
device= "cuda"
num_workers= 4
resume= False
resume_model= False
resume_optimizer= False
resume_scheduler= False
use_scheduler= True
lr_init= 0.001
batch_size= 32
test_batch_size= 32
num_epochs= 5
start_epoch= 0
patience= 5
lr_decay= 0.1
n_bins=800

dropout= False
