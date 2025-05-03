# define SCaM params

def __getattr__(name):

    if name in locals() : return locals()[name]
    else : raise ValueError(f"params (of SCaM) dont have {name=}.")


path = "./../output_simu"
folder_train = "train256"
folder_valid = "valid64"
folder_image = "image"
folder_spectrum = "spectrum"

out_path = "."
out_loss = "loss"
out_states ="states"
out_epoch = "epoch"

seed = 42
num_workers = 4
lr_init = 0.001
batch_size = 64
num_epochs = 5
# lr_decay = 0.1
n_bins = 800