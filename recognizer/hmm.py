import numpy as np
from recognizer.tools import viterbi,limLog

# default HMM
WORDS = {
    'name': ['sil', 'oh', 'zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine'],
    'size': [1, 3, 15, 12, 6, 9, 9, 9, 12, 15, 6, 9],
    'gram': [100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100],
}


class HMM:  

    words = {}
    logA= {}
    logPi={}

    def __init__(self, mode='adjust', words=WORDS):
        """
        Constructor of HMM class. Inits with provided structure words
        :param input: word of the defined HMM.
        """

        # as the transtion from any state to the start of 'oh'
        # was often falsy detected the probabilty of those transition 
        # is scaled down to counteract that
        if mode=='adjust':
            print("Adjusting hmm")
            scaling_factor_less_oh_transitions=0.01
        else:
            scaling_factor_less_oh_transitions=1
        
        null_prob=1/(len(WORDS['size'])+1)*scaling_factor_less_oh_transitions
        start_prob=(1-null_prob)/(len(WORDS['size'])+1)
        pi=np.zeros(sum(WORDS['size']))
        #word_transition_probability
        wtp=0.5

        
        word_starting_indices=[]
        word_ending_indices=[]
        summe=0
            
        counter=0
        for i in WORDS['size']:
            word_starting_indices.append(summe)
            summe+=i
            word_ending_indices.append(summe-1)
            
        # print("Start",word_starting_indices)
        # print("End",word_ending_indices)
        

        #filling pi at the word start states with probability
        wi=0
        for i in range(len(pi)):
            if i==word_starting_indices[wi]:
                if i==1:
                     pi[i]=null_prob
                else:
                    pi[i]=start_prob

                wi+=1
                if wi==len(word_starting_indices):
                    break
            else:
                pi[i]=0
            i+=1
        A=np.zeros(shape=(sum(WORDS['size']), sum(WORDS['size'])))
        
        #fill wordtransitions pi into A
        for i in range(A.shape[0]):
            if np.isin(i,word_ending_indices):
                A[i,:]=list(pi)

        for i in range(A.shape[0]):
            if np.isin(i,word_ending_indices):
                A[i][i]=start_prob
            else:
                A[i][i]=wtp
                if i<A.shape[0]:
                    A[i][i+1]=wtp

        #additionally to the general scale down
        # the worst cases are additionally scaled down
        #transitions: 'zero'->'oh', 'two'->'oh', 'four'->'oh'
        # if mode=="adjusted":
        #     A[0][1]=0.00001*A[0][1]
        #     A[18][1]=0.000001*A[18][1]
        #     A[36][1]=0.00001*A[18][1]
        
        self.logA=self.limLog(A)
        self.logPi=self.limLog(pi)
        print(self.logA.shape)
        
        self.words = words

    def limLog(self,x):
        """
        Log of x.

        :param x: numpy array.
        :return: log of x.
        """
        MINLOG = 1e-100
        return np.log(np.maximum(x, MINLOG))

    def get_num_states(self):
        """
        Returns the total number of states of the defined HMM.
        :return: number of states.
        """
        return sum(self.words['size'])

    def input_to_state(self, input):
        """
        Returns the state sequenze for a word.
        :param input: word of the defined HMM.
        :return: states of the word as a sequence.
        """
        if input not in self.words['name']:
            raise Exception('Undefined word/phone: {}'.format(input))

        # start index of each word
        start_idx = np.insert(np.cumsum(self.words['size']), 0, 0)

        # returns index for input's last state
        idx = self.words['name'].index(input) + 1

        start_state = start_idx[idx - 1]
        end_state = start_idx[idx]

        return [n for n in range(start_state, end_state) ]

    
    def getTranscription(self,stateSequence):
        words=[]

        
        test=[]
        start_found=False
        #
        wi=0
        #StateStartIndex
        si=0
        #StateEndIndex
        ei=0
        for s in stateSequence: 
            n=0
            if start_found==False:
                while n<sum(WORDS['size']):
                    #check if new word is starting
                    if s==sum(WORDS['size'][:n]):
                        #start of not sil detected
                        if n!=0:
                            wi=n
                            si=s
                            ei=s+WORDS['size'][n]-1
                            # print("Si:",si)
                            # print("Ei:",ei)
                            # print("n:",n)
                            start_found=True
                    n+=1
            else:
                if s==ei:
                    words.append(WORDS['name'][wi])
                    start_found=False
        return words

    
    def posteriors_to_transcription(self,posteriors):

        StateSequence, Pstar = viterbi(limLog(posteriors), self.logPi, self.logA)
        return self.getTranscription(StateSequence)