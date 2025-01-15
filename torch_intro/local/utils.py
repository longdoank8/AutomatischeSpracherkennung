import os
import numpy
import numpy as np
import json
from tqdm import tqdm
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import torch_intro.local.feature_extraction as fe

def get_metadata(datadir, sourcedatadir):
    """
    get_metadata() load the meta data in dictionary. 
    Input:
        datadir: <string> the folder containing the meta data
    Return: 
        The dictionaries of the training, dev and test set
    """

    # Dateien laden 
    train_file = os.path.join(datadir, "train.json")
    dev_file = os.path.join(datadir, "dev.json")
    test_file = os.path.join(datadir, "test.json")

    # read "train.json as traindict"
    with open(train_file, "r") as f:
        traindict = json.load(f)
     
    # read "dev.json as devdict"
    with open(dev_file, "r") as f:
        devdict = json.load(f)
    
    # read "test.json as testdict"
    with open(test_file, "r") as f:
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
        fbank_fmax=self.fbank_fmax, 
        #num_ceps=self.num_ceps, 
        #left_context=self.left_context, 
        #right_context=self.right_context
        )

        # transform numpy audio features to FloatTensor 
        audiofeat = torch.tensor(audiofeat, dtype=torch.float32) 

        # Crop the audio features, if the frame length longer than defined max_len                                           
        if audiofeat.shape[0] > self.max_len:    
            audiofeat = audiofeat[:self.max_len]

        # Move label to FloatTensor
        label = torch.FloatTensor([label_file_path])

        return audiofeat, label, filename