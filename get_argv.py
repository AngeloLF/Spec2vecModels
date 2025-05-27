from types import SimpleNamespace
import coloralf as c
import torch



def get_argv(argv, prog=None, correction=True, show=False):

    Args = SimpleNamespace()
    Args.save = True
    Args.is_pret, Args.prefixe = False, ""
    Args.from_pre = False
    Args.pre_train, Args.pre_lr, Args.pre_lr_str = None, None, None

    Args.device = "cpu"
    if prog in ["training"] : Args.device = "cuda"



    for arg in argv:

        if arg[:6] == "model=" : Args.model = arg[6:]
        if arg[:5] == "loss="  : Args.loss  = arg[5:]
        if arg[:6] == "train=" : Args.train = arg[6:]
        if arg[:6] == "valid=" : Args.valid = arg[6:]
        if arg[:5] == "test="  : Args.test  = arg[5:]

        if arg[:2] == "e="     : Args.epochs = int(arg[2:])
        if arg[:6] == "epoch=" : Args.epochs = int(arg[6:])
        if arg[:3] == "lr="    : Args.lr     = float(arg[3:])
        if arg[:3] == "lr="    : Args.lr_str = f"{float(arg[3:]):.0e}"
        
        if arg == "cpu"        : Args.device = "cpu"
        if arg == "gpu"        : Args.device = "cuda"
        if arg == "cuda"       : Args.device = "cuda"

        if arg == "nosave"     : Args.save = False

        if arg == "pre"        : 
            Args.is_pret = True
            Args.prefixe = "pre_"  
        if arg[:5] == "load="  : 
            load_pret = arg[5:] # <folder_pretrain>_<prelr>
            pre_train, pre_lr = load_pret.split("_")
            Args.from_pre = True
            Args.pre_train = pre_train if pre_train[:5] == "train" else f"train{pre_train}"
            Args.pre_lr = float(pre_lr)
            Args.pre_lr_str = f"{float(pre_lr):.0e}"


    
    if prog is not None:


        if "model" not in dir(Args) and prog in ["training", "apply", "analyse"]:

            print(f"{c.r}WARNING : model architecture is not define (model=<model_architecture>){c.d}")
            raise Exception("Model architecture not define")



        if "loss" not in dir(Args) and prog in ["training", "apply", "analyse"]:

            print(f"{c.r}WARNING : loss function is not define (loss=<loss_function>){c.d}")
            raise Exception("Loss function is not define")



        if "train" not in dir(Args) and prog in ["training", "apply", "analyse"]:

            print(f"{c.r}WARNING : train folder is not define (train=<train_folder>){c.d}")
            raise Exception("Train folder is not define")

        elif "train" in dir(Args) and correction and Args.train[:5] != "train" : Args.train = f"train{Args.train}"



        if "valid" not in dir(Args) and prog in ["training"]:

            print(f"{c.r}WARNING : valid folder is not define (valid=<valid_folder>){c.d}")
            raise Exception("Valid folder is not define")

        elif "valid" in dir(Args) and correction and Args.valid[:5] != "valid" : Args.valid = f"valid{Args.valid}"



        if "test" not in dir(Args) and prog in ["apply", "analyse"]:

            print(f"{c.r}WARNING : test folder is not define (test=<test_folder>){c.d}")
            raise Exception("test folder is not define")

        elif "test" in dir(Args) and correction and Args.test[:4] != "test" : Args.test = f"test{Args.test}"



        if "epochs" not in dir(Args) and prog in ["training"]:

            print(f"{c.r}WARNING : Epoch number is not define (e=<epochs_number> or epoch=<epochs_number){c.d}")
            raise Exception("Epoch number is not define")



        if "lr" not in dir(Args) and prog in ["training", "apply", "analyse"]:

            print(f"{c.r}WARNING : learning rate is not define (lr=<learning_rate>){c.d}")
            raise Exception("Learning rate is not define")



    if show : show_Args(Args)

    return Args




def show_Args(args):

    atts = [a for a in dir(args) if a[:2] != "__"]

    print(f"\n{c.y}Args attributs :{c.d}")

    for att in atts:

        print(f"{c.ly}{att}{c.d} : {args.__getattribute__(att)}")

    print(f"")



def get_device(args):

    if args.device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        if args.device == "cuda" : print(f"{c.r}WARNING : GPU is not available for torch ... device turn to CPU ... ")
        device = torch.device("cpu")

    return device


if __name__ == "__main__":

    import sys

    args = get_argv(sys.argv[2:], sys.argv[1], show=True)