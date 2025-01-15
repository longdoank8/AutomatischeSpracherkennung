import os
import numpy
import numpy as np
import json
from tqdm import tqdm
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import torch_intro.local.feature_extraction as fe


def get_metadata(datadir):
    """
    get_metadata() load the meta data in dictionary. 
    Input:
        datadir: <string> the folder containing the meta data
    Return: 
        The dictionaries of the training, dev and test set
    """

    # Dateien laden 
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

    audio_feat_sequence = []
    label_sequence = []
    filename_sequence = []

    for seq in sequences:
  
        audio_feat_sequence.append(seq[0])  
        label_sequence.append(seq[1])      
        filename_sequence.append(seq[2])   

    # Zero-Padding of Audio-Features
    audio_feat_sequence = pad_sequence(audio_feat_sequence, batch_first=True)

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


    def _get_keys(self, datadict):
        keys = list(datadict.keys())
        return keys

    def __len__(self):
        return len(self.datakeys)

    def __getitem__(self, index):

        # The audio sample file name
        filename = self.datakeys[index] 

        # Get audio sample path
        audio_file_path = self.datadict[filename]["sampledir"]
        
        # Get label
        label_file_path = self.datadict[filename]["label"]
        
        # Extract audio features 
        audiofeat = fe.compute_features(audio_file_path, 
        feature_type=self.feature_type,
        window_size=self.window_size, 
        hop_size=self.hop_size, 
        n_filters=self.n_filters, 
        fbank_fmin=self.fbank_fmin, 
        fbank_fmax=self.fbank_fmax
        )

        # transform numpy audio features to FloatTensor 
        audiofeat = torch.FloatTensor(audiofeat) 

        # Move label to FloatTensor
        label = torch.FloatTensor([label_file_path])

        return audiofeat, label, filename