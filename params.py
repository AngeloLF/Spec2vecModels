# define SCaM params

def __getattr__(name):

    if name in locals() : return locals()[name]
    else : raise ValueError(f"params (of Spec2vecModels) dont have {name=}.")

path = "./results/output_simu"

out_path = "./results"
out_dir = "Spec2vecModels_Results"
out_loss = "loss"
out_loss_mse = "loss_mse"
out_loss_chi2 = "loss_chi2"
out_states = "states"
out_prestates = "pre_states" 
out_epoch = "epoch"
out_divers = "divers"

seed = 42
num_workers = 4
lr_default = 0.001
batch_size = 64
num_epochs = 2
n_bins = 800
Csigma_chi2 = 12.