import os
import numpy
import numpy as np
import json
from tqdm import tqdm
import torch
from torch.utils.data import Dataset
import random
import recognizer.feature_extraction as fe
import recognizer.hmm as HMM
import recognizer.tools as tools
from scipy.io import wavfile

def get_data(datadir):
    """
    get_metadata() load the meta data in dictionary. 
    Input:
        datadir: <string> the folder containing the meta data
    Return: 
        The dictionaries of the training, dev and test set
    """


    f = open(os.path.join(datadir, "train.json"))
    traindict = json.load(f)

    f = open(os.path.join(datadir, "dev.json"))
    devdict = json.load(f)

    f = open(os.path.join(datadir, "test.json"))
    testdict = json.load(f)

    return traindict, devdict, testdict


def padding(sequences):
    '''
    Wendet Zero-Padding auf die Merkmale eines Batches an, sodass alle Merkmalssequenzen die gleiche Länge haben.
    Input:
        sequences <list>: Liste von Tuples (audio_feat, label, filename) mit
                          - audio_feat: Tensor der Audio-Merkmale
                          - label: Tensor des Labels
                          - filename: Name der Audiodatei
    Return:
        audio_feat_sequence <torch.FloatTensor>: Zero-gepaddete Audio-Merkmale.
        label_sequence <torch.FloatTensor>: Labels als Tensor.
        filename_sequence <list>: Dateinamen als Liste.
    '''
    # Listen zum Speichern der Audio-Features, Labels und Dateinamen
    audio_feat_sequence = []
    label_sequence = []
    filename_sequence = []

    for seq in sequences:
        # Extrahiere die Audio-Features, Labels und Dateinamen
        audio_feat_sequence.append(seq[0])  # Audio-Features
        label_sequence.append(seq[1])      # Labels
        filename_sequence.append(seq[2])   # Dateinamen

    # Zero-Padding der Audio-Features
    # pad_sequence erwartet eine Liste von Tensors, nicht von Listen oder zusätzlichen Dimensionen
    audio_feat_sequence = pad_sequence(audio_feat_sequence, batch_first=True)

    # Labels in einen Tensor umwandeln
    label_sequence = torch.stack(label_sequence)

    return audio_feat_sequence, label_sequence, filename_sequence

class Dataloader(Dataset):  # For instantiating train, validation and test dataset
    def __init__(self, datadict, feat_params):
        super(Dataloader).__init__()
        self.datakeys = self._get_keys(datadict)
        self.datadict = datadict                        # The Python-dictionary which imported from the json file
        self.window_size = feat_params[0]               # The hyper-parameters for feature extraction
        self.hop_size = feat_params[1]
        self.feature_type = feat_params[2]
        self.n_filters = feat_params[3]
        self.fbank_fmin = feat_params[4]
        self.fbank_fmax = feat_params[5]
        self.max_len = feat_params[6]
        self.num_ceps = feat_params[7]
        self.left_context = feat_params[8]
        self.right_context = feat_params[9]
        self.hmm = HMM.HMM()


    def _get_keys(self, datadict):
        keys = list(datadict.keys())
        return keys

    def __len__(self):
        return len(self.datakeys)

    def __getitem__(self, index):
        filename = self.datakeys[index]
        audio_file_path = self.datadict[filename]["audiodir"]
        label_file_path = self.datadict[filename]["targetdir"]
        #word_file_path = self.datadict[filename]["audiodir"].replace("wav", "lab")
        
        # Extract audio features 
        audiofeat = fe.compute_features_with_context(audio_file_path, 
        feature_type=self.feature_type,
        window_size=self.window_size, 
        hop_size=self.hop_size, 
        n_filters=self.n_filters, 
        fbank_fmin=self.fbank_fmin, 
        fbank_fmax=self.fbank_fmax, 
        num_ceps=self.num_ceps, 
        left_context=self.left_context, 
        right_context=self.right_context
        )

        # transform numpy audio features to FloatTensor 
        audiofeat = torch.FloatTensor(audiofeat) 

        sampling_rate = wavfile.read(audio_file_path)[0]
    
        # calulate sample size of the dft window
        window_size_samples = tools.dft_window_size(self.window_size, sampling_rate)

        # calculate hop size in samples from hop size in seconds
        hop_size_samples = tools.sec_to_samples(self.hop_size, sampling_rate)
        
        label = torch.FloatTensor(tools.praat_file_to_target(label_file_path, sampling_rate=sampling_rate, window_size_samples = window_size_samples, hop_size_samples = hop_size_samples, hmm = self.hmm))
        #words = [word.lower() for word in open(word_file_path).read().split()]

        return audiofeat, label, filename