# define SCaM params

def __getattr__(name):

    if name in locals() : return locals()[name]
    else : raise ValueError(f"params (of SCaM) dont have {name=}.")

path = "./results/output_simu"
folder_image = "image"
folder_spectrum = "spectrum"

out_path = "./results"
out_dir = "Spec2vecModels_Results"
out_loss = "loss"
out_states ="states"
out_epoch = "epoch"

seed = 42
num_workers = 4
lr_init = 1e-4
batch_size = 64
num_epochs = 2
n_bins = 800

lasso = 1e-3
ridge = 1e-3