import math
import numpy as np
from praatio import tgio

def sec_to_samples(x, sampling_rate):

    return int(x * sampling_rate)
    

def next_pow2(x):

    return math.ceil(math.log2(abs(x)))
    

# next pow2 rundet immer auf die naechst hoehere potenz
# bsp: fuer 300 => 2^8 ist 256
# man will aber eine potenz finden die groesser 300 ist
# hoehere ist 2^9 = 500...

# gibt die 2^9 zurueck, da die dft besser mit den zweierpotenzen arbeitet, statt einer ganzen zahl
def dft_window_size(x, sampling_rate):

    samples = sec_to_samples(x, sampling_rate)
    b = 2 ** next_pow2(samples)
    return b 
    pass

# window_size gibt die abtastrate an zb 512 samples/abtastraten
# ein frame hat dann 512 samples
# hop_size ist die einheit, in der das ganze signal nach und nach in frames abgetastet wird
# man nimmt hop_size, damit sogenannte overlaps entstehen, die dann praeziser sind, da mehrmals die gleichen abschnitte/overlaps 
# vom signal genommen werden
def get_num_frames(signal_length_samples, window_size_samples, hop_size_samples):

    overlap = window_size_samples - hop_size_samples

    num_frames = (signal_length_samples - overlap) / hop_size_samples

    return math.ceil(num_frames) # mit math bib wird zu math.ceil(num_frames)
    pass

def hz_to_mel(x):
    # berechnet den Wert der Mel-Skala für den entsprechenden Frequenzwert x
    return 2595 * np.log10(1 + x/700 )

def mel_to_hz(x):
    # Konvertiert eine Frequenz x (in Mel) zurück in Hz
    return 700 * (10**(x / 2595) - 1 )


def sec_to_frame(x, sampling_rate, hop_size_samples):
    """
    Converts time in seconds to frame index.

    :param x:  time in seconds
    :param sampling_rate:  sampling frequency in hz
    :param hop_size_samples:    hop length in samples
    :return: frame index
    """
    return int(np.floor(sec_to_samples(x, sampling_rate) / hop_size_samples))


def divide_interval(num, start, end):
    """
    Divides the number of states equally to the number of frames in the interval.

    :param num:  number of states.
    :param start: start frame index
    :param end: end frame index
    :return starts: start indexes
    :return end: end indexes
    """
    interval_size = end - start
    # gets remainder 
    remainder = interval_size % num
    # init sate count per state with min value
    count = [int((interval_size - remainder)/num)] * num
    # the remainder is assigned to the first n states
    count[:remainder] = [x + 1 for x in count[:remainder]]
    # init starts with first start value
    starts = [start]
    ends = [] 
    # iterate over the states and sets start and end values
    for c in count[:-1]:
        ends.append(starts[-1] + c)
        starts.append(ends[-1])

    # set last end value
    ends.append(starts[-1] + count[-1])

    return starts, ends


def praat_file_to_target(praat_file, sampling_rate, window_size_samples, hop_size_samples, hmm):
    """
    Reads in praat file and calculates the word-based target matrix.

    :param praat_file: *.TextGrid file.
    :param sampling_rate: sampling frequency in hz
    :param window_size_samples: window length in samples
    :param hop_size_samples: hop length in samples
    :return: target matrix for DNN training
    """
    # gets list of intervals, start, end, and word/phone
    intervals, min_time, max_time = praat_to_interval(praat_file)

    # gets dimensions of target
    max_sample = sec_to_samples(max_time, sampling_rate)
    num_frames = get_num_frames(max_sample, window_size_samples, hop_size_samples)
    num_states = hmm.get_num_states()

    # init target with zeros
    target = np.zeros((num_frames, num_states))

    # parse intervals
    for interval in intervals:
        # get state index, start and end frame
        states = hmm.input_to_state(interval.label)
        start_frame = sec_to_frame(interval.start, sampling_rate, hop_size_samples)
        end_frame = sec_to_frame(interval.end, sampling_rate, hop_size_samples)

        # divide the interval equally to all states
        starts, ends = divide_interval(len(states), start_frame, end_frame)

        # assign one-hot-encoding to all segments of the interval
        for state, start, end in zip(states, starts, ends):    
            # set state from start to end to 1
            target[start:end, state] = 1

    # find all columns with only zeros...
    zero_column_idxs = np.argwhere(np.amax(target, axis=1) == 0)
    # ...and set all as silent state
    target[zero_column_idxs, hmm.input_to_state('sil')] = 1

    return target


def praat_to_interval(praat_file):
    """
    Reads in one praat file and returns interval description.

    :param praat_file: *.TextGrid file path

    :return itervals: returns list of intervals, 
                        containing start and end time and the corresponding word/phone.
    :return min_time: min timestamp of audio (should be 0)
    :return max_time: min timestamp of audio (should be audio file length)
    """
    # read in praat file (expects one *.TextGrid file path)
    tg = tgio.openTextgrid(praat_file)

    # read return values
    itervals = tg.tierDict['words'].entryList
    min_time = tg.minTimestamp
    max_time = tg.maxTimestamp

    # we will read in word-based
    return itervals, min_time, max_time

def viterbi( logLike, logPi, logA ):
    
  logLike = np.array(logLike)  
  logPi = np.array(logPi) 
  logA = np.array(logA) 

  phi_t = []
  psi_t = [[-1] * logA.shape[0]]

  phi_t.append([logPi + logLike[0, :]])

  for t in range(1, logLike.shape[0]):
    phi_j = []
    psi_j = []
    for j in range(logLike.shape[1]):
      phi_j.append(np.max(phi_t[t-1] + logA[:, j]) + logLike[t, j])
      psi_j.append(np.argmax(phi_t[t-1] + logA[:, j]))
    phi_t.append(phi_j)
    psi_t.append(psi_j)

  pstar = np.max(phi_t[-1])
  
  tmp_state = np.argmax(phi_t[-1])
  stateSequence = [tmp_state]
  t = logLike.shape[0] - 1
  while(t != 0):
    tmp_state = psi_t[t][tmp_state]
    stateSequence.insert(0, tmp_state)
    t-= 1

  return stateSequence, pstar


def limLog(x):
    """
    Log of x.

    :param x: numpy array.
    :return: log of x.
    """
    epsi = np.full(x.shape, np.nextafter(0, 1))
    x = np.where(x <= 0, epsi, x)
    return np.log(x) 