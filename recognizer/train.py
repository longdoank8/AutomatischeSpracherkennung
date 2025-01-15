import os, sys
import json
import argparse
import numpy as np
import torch
from tqdm import tqdm
from recognizer.model import *
from recognizer.utils import *
import torch.nn.functional as F
import matplotlib.pyplot as plt

def wav_to_posteriors(model, audio_file, parameters, device):
    model.eval()
    


    audiofeat = fe.compute_features_with_context(audio_file, feature_type=parameters[2],
                                                window_size=parameters[0],
                                                hop_size=parameters[1],
                                                n_filters=parameters[3],
                                                fbank_fmin=parameters[4],
                                                fbank_fmax=parameters[5],
                                                num_ceps=parameters[7],
                                                left_context=parameters[8],
                                                right_context=parameters[9])

    audiofeat = torch.FloatTensor(audiofeat).unsqueeze(0).to("cuda:0" if device == "gpu" else "cpu")

    output = model(audiofeat).detach().squeeze().to("cuda:0" if device == "gpu" else "cpu")

    output = F.softmax(output)
    output = output.to("cpu")
    return output

def train(dataset, model, odim, device, epoch, optimizer=None, criterion=None):
    """
    train() trains the model. 
    Input:
        dataset: dataset used for training.
        model: model to be trained.
        optimizer: optimizer used for training.
        criterion: defined loss function
    Return: 
        the performance on the training set and the trained model.
    """
    # move data onto gpu, if gpu available
    # zero the parameter gradients
    # using model compute posterior probabilities
    # compute loss value
    # compute accuracy

        
    # bring model into training mode
    model.train()

    # Set device to cuda if available
    device_ = torch.device("cuda:0" if device == "gpu" else "cpu")
    accuracy = []
    for batch in tqdm(dataset, desc="Training epoch " + str(epoch)):
        inputs, labels, _ = batch

        inputs = inputs.to(device_)
        labels = labels.to(device_)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Compute posterior probabilities
        y = model(inputs).to(device_)
        # return the index of best classification
        pred_max_index = torch.argmax(y.squeeze(), axis = 1).to(device_)
        labels_max_index = torch.argmax(labels.squeeze(), axis = 1).to(device_)
        # sum of all correct / all
        batch_accuracy = (pred_max_index == labels_max_index).sum() / labels_max_index.shape[0]
        accuracy.append(batch_accuracy)

        # Compute loss
        loss = criterion(y.squeeze(), labels.squeeze())
        # Compute gradients
        loss.backward()
        # Adjust learning weights
        optimizer.step()
    avg_accuracy = sum(accuracy) / len(accuracy)
    return avg_accuracy, model



def evaluation(dataset, model, odim, device, epoch):
    """
    evaluation() is used to evaluate the model. 
    Input:
        dataset: the dataset used for evaluation.
        model: the trained model.
    Return: 
        the accuracy on the given dataset, the predictions saved in dictionary and the model.
    """
                                                            # bring model into evaluation mode
                                                            # traverse each batch of samples
                                                            # move data onto gpu if gpu available

                                                            # using trained model, compute posterior probabilities
                                                            # compute accuracy
    model.eval()
    device_ = torch.device("cuda:0" if device == "gpu" else "cpu")
    accuracy = []
    for batch in tqdm(dataset, desc="Evaluation epoch " + str(epoch)):
        inputs, labels, _ = batch

        inputs = inputs.to(device_)
        labels = labels.to(device_)
        y = model(inputs).to(device_)

        # return the index of best classification
        pred_max_index = torch.argmax(y.squeeze(), axis = 1).to(device_)
        labels_max_index = torch.argmax(labels.squeeze(), axis = 1).to(device_)
        # sum of all correct / all
        batch_accuracy = (pred_max_index == labels_max_index).sum() / labels_max_index.shape[0]
        accuracy.append(batch_accuracy)

    accuracy = sum(accuracy) / len(accuracy)

    return accuracy, model

def run(config, args, datadicts=None):
    """
    run() trains and evaluates the model over given number of epochs.
    Input:
        config: the defined hyperparameters
        datadicts: the dictionary containing the meta-data for training, dev and test set.
    """
    traindict, devdict, testdict = datadicts  
    # Parameters for feature extraction
    feat_params = [config["window_size"], config["hop_size"],
                config["feature_type"], config["n_filters"],
                config["fbank_fmin"], config["fbank_fmax"],
                config["max_frame_len"], config["num_ceps"], 
                config["left_context"], config["right_context"]]


    # Create 3 datasets from given training, dev and test meta-data
    train_dataset = Dataloader(traindict, feat_params)
    dev_dataset = Dataloader(devdict, feat_params)
    test_dataset = Dataloader(testdict, feat_params)

    print(len(train_dataset))
    print(len(dev_dataset))
    print(len(test_dataset))


    # Define loss function, model and optimizer
    criterion = torch.nn.CrossEntropyLoss() 

    idim = config["num_ceps"] * 3 * (config["right_context"] + config["left_context"] + 1)  


    # Define loss function, model and optimizer
    criterion = torch.nn.CrossEntropyLoss()                                                  # Binary cross entropy as loss function.
    model = DNN_L_Model(idim=idim, odim=train_dataset.hmm.get_num_states(), hidden_dim=512)        # Initial model       # Initial model
    model = model.to(config["device"])  # move model to gpu, if gpu available
                                                
    optimizer = torch.optim.Adam(model.parameters(),                                # Initialize an optimizer
                                 lr=config["lr"]
                                 )

    # Pre-loading dataset
    data_loader_train = torch.utils.data.DataLoader(train_dataset,                  # Create dataset
                                                    shuffle=True,                   # Randomly shuffle if shuffle=True
                                                    batch_size=config["batch_size"],# Defined batch size
                                                    num_workers=config["NWORKER"],  # A positive integer will turn on multi-process data loading
                                                    drop_last=False)               # If drop_last=True, the data loader will drop the last batch if there are not enough remaining samples for a batch
                                                 

    data_loader_dev = torch.utils.data.DataLoader(dev_dataset, shuffle=True,
                                                      batch_size=config["batch_size"],
                                                      num_workers=config["NWORKER"])

    best_model = None
    best_eval = 0.0                                                 

    model_output_path = os.path.join(args.savedir, "model")
    for epoch in range(config["epochs"]):  # loop over the dataset multiple times
        # Train model on training set
        odim = train_dataset.hmm.get_num_states()
        trainscore, model = train(data_loader_train,
                                     model,
                                     odim,
                                     config["device"],
                                     epoch,
                                     optimizer=optimizer,
                                     criterion=criterion)
        # Evaluate trained model on dev set
        print("Train epoch ", str(epoch), " accuracy: ", trainscore)
        
        # print("evalscore: ", evalscore)
        evalscore, model = evaluation(data_loader_dev, model, odim,
                                     config["device"], epoch)
        print("Evaluation epoch ", str(epoch), " accuracy: ", evalscore)
        if evalscore > best_eval:
            torch.save(model.state_dict(), os.path.join(model_output_path, "best_model.pth"))
            best_eval = evalscore

        torch.save(model.state_dict(), os.path.join(model_output_path, "model_epoch" + str(epoch) +".pth"))

    model.load_state_dict(torch.load(os.path.join(model_output_path, "best_model.pth")))
    # label_file_path = os.path.join(args.sourcedatadir, "TEST/TextGrid/TEST-WOMAN-BF-7O17O49A.TextGrid")
    audiofeat, label, filename = test_dataset[0]
    audio_file = os.path.join(args.sourcedatadir, "TEST/wav", filename + ".wav")
    output = wav_to_posteriors(model, audio_file, feat_params, config["device"])


    # Create subplots
    fig, axs = plt.subplots(2, 1, figsize=(8, 6))

    # Plot Ground-Truth Labels
    im = axs[0].imshow(label.T, origin="lower")
    axs[0].set_xlabel("Frames")
    axs[0].set_ylabel("HMM states")
    axs[0].set_title("GT Labels")

    # Plot A-posteriori Wahrscheinlichkeiten
    axs[1].imshow(output.T, origin="lower")
    axs[1].set_xlabel("Frames")
    axs[1].set_ylabel("HMM states")
    axs[1].set_title("A-post. Wahrscheinlichkeiten")

    # Create a colorbar for both subplots
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label('Intensity')

    plt.tight_layout()
    plt.show()





