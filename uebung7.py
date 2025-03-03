import argparse
from recognizer.train import run
from recognizer.utils import *
import torch
import os

import random
SEED=42
random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
def get_args():
    parser = argparse.ArgumentParser()

    # get arguments from outside
    parser.add_argument('--sourcedatadir', default='./data/TIDIGITS-ASE', type=str, help='Dir saves the datasource information')
    parser.add_argument('--datasdir', default='./dataset', type=str, help='Dir saves the datasource information')
    parser.add_argument('--savedir', default='./trained', type=str, help='Dir to save trained model and results')
    # parser.add_argument('--resume', default='', type=str, help='Path to trained model') 
    # for later
    parser.add_argument('--nepochs', default=1, type=int, help='Number of epochs to train for') 
    parser.add_argument('--lr', default=0.001, type=float, help='Learning rate')
    parser.add_argument('--hidden-dim', default=512, type=int, help='Hidden neurons number')
    parser.add_argument('--model-l', action="store_true", help='Use flag to create larger model')


    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_args()
    print("Arguments:")
    [print(key, val) for key, val in vars(args).items()]

    sourcedatadir = args.sourcedatadir
    datasetdir = args.datasdir
    savedir = args.savedir

    # If GPU on device available, use GPU to train the model

    if torch.cuda.is_available() == True:
         device = 'gpu'     # use GPU
         print("Use Device = GPU")
    else:
         device = 'cpu'      # use CPUS
         print("Use Device = CPU")



    # Create folders to save the trained models and evaluation results
    modeldir = os.path.join(savedir, 'model')
    resultsdir = os.path.join(savedir, 'results')
    for makedir in [modeldir, resultsdir, datasetdir]:
        if not os.path.exists(makedir):
            os.makedirs(makedir)

    # Load meta data as dictionary
    traindict, devdict, testdict = get_data(args.datasdir)
    # Config audio paths
    for dict in [traindict, devdict, testdict]:
        for k, v in dict.items():
            dict[k]["audiodir"] = dict[k]["audiodir"].replace('./TIDIGITS-ASE', sourcedatadir)
            dict[k]["targetdir"] = dict[k]["targetdir"].replace('./TIDIGITS-ASE', sourcedatadir)


    # Configure hyperparameters
    config = {
        "NWORKER": 10,
        "device": device,
        "lr": 0.001,
        "batch_size": 1,
        "epochs": 3, 
        "window_size": 25e-3,
        "hop_size": 10e-3,
        "feature_type": 'MFCC_D_DD',
        "n_filters": 40,
        "fbank_fmin": 0,
        "fbank_fmax": 8000,
        "hidden_size": 512,
        "max_frame_len": 768,
        "num_ceps": 13,
        "left_context": 10,
        "right_context": 10,
        "modeldir": modeldir,
        "resultsdir": resultsdir
    }

    run(config,  args, datadicts=[traindict, devdict, testdict])




