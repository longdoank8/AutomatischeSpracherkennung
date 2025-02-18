import recognizer.hmm as HMM
import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":

    # default HMM
    hmm = HMM.HMM()

    statesequence = [0, 0, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 0, 31, 32, 33, 34, 35, 36, 0, 37, 38, 39, 40, 41, 42, 43, 44, 45, 0]

    words = hmm.getTranscription(statesequence)
    print(words) # ['ONE', 'TWO', 'THREE']

    plt.imshow(np.exp(hmm.logA))
    plt.xlabel('nach Zustand j')
    plt.ylabel('von Zustand i')
    plt.colorbar()
    plt.show()
