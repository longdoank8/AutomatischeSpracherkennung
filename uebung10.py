import argparse
import torch 
import os
import matplotlib.pyplot as plt
from recognizer.utils import *
from recognizer.train import wav_to_posteriors
from recognizer.model import *

def test_model(datadir, hmm, model, parameters, dataset):
    N = D = I = S = 0

    _, _, testdict = get_data(dataset)

    for audio_name in tqdm(testdict):
        audio_file = os.path.join(datadir, "TEST/wav", audio_name + ".wav")
        words_file = audio_file.replace("wav", "lab")
        posteriors = wav_to_posteriors(model, audio_file, parameters, device)
        our_words = hmm.posteriors_to_transcription(posteriors)

        label_words = [word.lower() for word in open(words_file).read().split()]

        N_, D_, I_, S_ = tools.needlemann_wunsch(label_words, our_words) 
        N += N_
        D += D_
        I += I_
        S += S_
        print("---------------------------------------------------------")
        print(audio_file)
        print("REF: ", label_words)
        print("OUT: ", our_words)
        print("I: ", I, " D: ", D, " S: ", S, " N: ", N)
        print("Current Total WER: ", 100*((D+I+S)/N))
    final_WER=100*((D+I+S)/N)
    return final_WER



def get_args():
    parser = argparse.ArgumentParser()
    # get arguments from outside
    parser.add_argument('--sourcedatadir', default='./data/TIDIGITS-ASE', type=str, help='Dir saves the datasource information')
    parser.add_argument('--datasdir', default='./dataset', type=str, help='Dir saves the datasource information')
    parser.add_argument('--savedir', default='./trained', type=str, help='Dir to save trained model and results')
    parser.add_argument('--model', default='./trained/model/best_model.pth', type=str, help='Dir to save trained model and results')
    parser.add_argument('--hidden-dim', default=512, type=int, help='Hidden neurons number')
    parser.add_argument('--hmm-adjust', action="store_true", help='Use flag to adjust (make smaller) probability for "oh" word')
    parser.add_argument('--model-l', action="store_true", help='Use flag to create larger model')

    args = parser.parse_args()
    return args

if __name__ == "__main__":

    # parse arguments
    # data directory, e.g., /media/public/TIDIGITS-ASE
    # call:
    # python uebung10.py <data/dir>
    # e.g., python uebung11.py /media/public/TIDIGITS-ASE
    args = get_args()
    print("Arguments:")
    [print(key, val) for key, val in vars(args).items()]

    datadir = args.sourcedatadir
    savedir = args.savedir

    if torch.cuda.is_available() == True:
        device = 'gpu'  # use GPU
        print("Use Device = GPU")
    else:
        device = 'cpu'  # use CPUS
        print("Use Device = CPU")

    # parameters for the feature extraction
    parameters = {'window_size': 25e-3,
        'hop_size': 10e-3,
        'feature_type': 'MFCC_D_DD',
        'n_filters': 40,
        'fbank_fmin': 0,
        'fbank_fmax': 8000,
        'num_ceps': 13,
        "max_frame_len": 768,
        'left_context': 10,
        'right_context': 10}

    # default HMM
    mode = "adjust" if args.hmm_adjust else "default"
    hmm = HMM.HMM(mode=mode)

    # 1.) Test mit vorgegebenen Daten
    # die Zustandswahrscheinlichkeiten passend zum HMM aus UE6
    posteriors = np.load('data/TEST-WOMAN-BF-7O17O49A.npy')

    # Transkription für die vorgegebenen Wahrscheinlichkeiten
    words = hmm.posteriors_to_transcription(posteriors)
    print('Given posteriori OUT: {}'.format(words))            # OUT: [’SEVEN’, ’OH’, ’ONE’, ’SEVEN’, ’OH’, ’FOUR’, ’NINE’]

    #####################################################################
    # in Übung7 trainiertes DNN Model name
    model_name = args.model
    # Model Pfad
    model_dir = os.path.join(savedir, 'model', model_name + '.pkl')
    # Laden des DNNs
    feat_params = [parameters["window_size"], parameters["hop_size"],
                parameters["feature_type"], parameters["n_filters"],
                parameters["fbank_fmin"], parameters["fbank_fmax"],
                parameters["max_frame_len"], parameters["num_ceps"], 
                parameters["left_context"], parameters["right_context"]]
    idim = parameters["num_ceps"] * 3 * (parameters["right_context"] + parameters["left_context"] + 1)

    if(args.model_l):
        model = DNN_L_Model(idim=idim, odim=hmm.get_num_states(), hidden_dim=args.hidden_dim) 
    else:
        model = DNN_Model(idim=idim, odim=hmm.get_num_states(), hidden_dim=args.hidden_dim) 

    if(device == "gpu"):
        model.to("cuda:0").load_state_dict(torch.load(args.model))
    else:
        model.to("cpu").load_state_dict(torch.load(args.model, map_location=torch.device("cpu")))
    model.eval()


    #####################################################################

    # Beispiel wav File
    test_audio = os.path.join(args.sourcedatadir, 'TEST/wav/TEST-WOMAN-BF-7O17O49A.wav')
    # TODO
    # Hier bitte den eigenen Erkenner laufen lassen und das Ergebnis vergleichen
    own_posteriors = wav_to_posteriors(model, test_audio, feat_params, device)
    own_words = hmm.posteriors_to_transcription(own_posteriors)
    print('OUR posteriori OUT: {}'.format(own_words))
    # print('OUT: {}'.format(words))  # OUT: [’SEVEN’, ’OH’, ’ONE’, ’SEVEN’, ’OH’, ’FOUR’, ’NINE’]

    # test DNN
    wer = test_model(datadir, hmm, model, feat_params, args.datasdir)
    print('--' * 40)
    print("Total WER: {}".format(wer))
