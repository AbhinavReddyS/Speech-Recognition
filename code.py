import glob
import os
import wer
import observation_model
import openfst_python as fst
import math
from subprocess import check_call
from IPython.display import Image
from timeit import default_timer as timer
from datetime import datetime

class MyViterbiDecoder:
    
    NLL_ZERO = 1e10  # define a constant representing -log(0).  This is really infinite, but approximate
                     # it here with a very large number

    new_lst = []
    
    def __init__(self, f, audio_file_name):
        """Set up the decoder class with an audio file and WFST f
        """
        self.om = observation_model.ObservationModel()
        self.f = f
        
        if audio_file_name:
            self.om.load_audio(audio_file_name)
        else:
            self.om.load_dummy_audio()
        
        self.initialise_decoding()

        
    def initialise_decoding(self):
        """set up the values for V_j(0) (as negative log-likelihoods)
        
        """
        
        self.V = []   # stores likelihood along best path reaching state j
        self.B = []   # stores identity of best previous state reaching state j
        self.W = []   # stores output labels sequence along arc reaching j - this removes need for 
                      # extra code to read the output sequence along the best path
        
        for t in range(self.om.observation_length()+1):
            self.V.append([self.NLL_ZERO]*self.f.num_states())
            self.B.append([-1]*self.f.num_states())
            self.W.append([[] for i in range(self.f.num_states())])  #  multiplying the empty list doesn't make multiple
        
        # The above code means that self.V[t][j] for t = 0, ... T gives the Viterbi cost
        # of state j, time t (in negative log-likelihood form)
        # Initialising the costs to NLL_ZERO effectively means zero probability    
        
        # give the WFST start state a probability of 1.0   (NLL = 0.0)
        self.V[0][self.f.start()] = 0.0
        
        # some WFSTs might have arcs with epsilon on the input (you might have already created 
        # examples of these in earlier labs) these correspond to non-emitting states, 
        # which means that we need to process them without stepping forward in time.  
        # Don't worry too much about this!  
        self.traverse_epsilon_arcs(0)        
        
    def traverse_epsilon_arcs(self, t):
        """Traverse arcs with <eps> on the input at time t
        
        These correspond to transitions that don't emit an observation
        
        """
        
        states_to_traverse = list(self.f.states()) # traverse all states
        while states_to_traverse:
            
            # Set i to the ID of the current state, the first 
            # item in the list (and remove it from the list)
            i = states_to_traverse.pop(0)   
        
            # don't bother traversing states which have zero probability
            if self.V[t][i] == self.NLL_ZERO:
                    continue
        
            for arc in self.f.arcs(i):
                
                if arc.ilabel == 0:     # if <eps> transition
                  
                    j = arc.nextstate   # ID of next state  
                
                    if self.V[t][j] > self.V[t][i] + float(arc.weight):
                        
                        # this means we've found a lower-cost path to
                        # state j at time t.  We might need to add it
                        # back to the processing queue.
                        self.V[t][j] = self.V[t][i] + float(arc.weight)
                        
                        # save backtrace information.  In the case of an epsilon transition, 
                        # we save the identity of the best state at t-1.  This means we may not
                        # be able to fully recover the best path, but to do otherwise would
                        # require a more complicated way of storing backtrace information
                        self.B[t][j] = self.B[t][i] 
                        
                        # and save the output labels encountered - this is a list, because
                        # there could be multiple output labels (in the case of <eps> arcs)
                        if arc.olabel != 0:
                            self.W[t][j] = self.W[t][i] + [arc.olabel]
                        else:
                            self.W[t][j] = self.W[t][i]
                        
                        if j not in states_to_traverse:
                            states_to_traverse.append(j)


    def forward_step(self, t):
          
        for i in self.f.states():

                if not self.V[t-1][i] == self.NLL_ZERO:   # no point in propagating states with zero probability
                    
                    for arc in self.f.arcs(i):
                        
                        if arc.ilabel != 0: # <eps> transitions don't emit an observation
                            j = arc.nextstate
                            tp = float(arc.weight)  # transition prob
                            ep = -self.om.log_observation_probability(self.f.input_symbols().find(arc.ilabel), t)  # emission negative log prob
                            prob = tp + ep + self.V[t-1][i] # they're logs
                            if prob < self.V[t][j]:
                                self.V[t][j] = prob
                                self.B[t][j] = i
                                
                                # store the output labels encountered too
                                if arc.olabel !=0:
                                    self.W[t][j] = [arc.olabel]
                                else:
                                    self.W[t][j] = []
        

    def forward_step_beam_search(self, t):
          
        for i in self.f.states():

            if t==1 or  (i in self.new_lst) :

                if not self.V[t-1][i] == self.NLL_ZERO:   # no point in propagating states with zero probability
                    
                    for arc in self.f.arcs(i):
                        
                        if arc.ilabel != 0: # <eps> transitions don't emit an observation
                            j = arc.nextstate
                            tp = float(arc.weight)  # transition prob
                            ep = -self.om.log_observation_probability(self.f.input_symbols().find(arc.ilabel), t)  # emission negative log prob
                            prob = tp + ep + self.V[t-1][i] # they're logs
                            if prob < self.V[t][j]:
                                self.V[t][j] = prob
                                self.B[t][j] = i
                                
                                # store the output labels encountered too
                                if arc.olabel !=0:
                                    self.W[t][j] = [arc.olabel]
                                else:
                                    self.W[t][j] = []
    
        
        self.new_lst = sorted(range(len(self.V[t])), key=lambda k: self.V[t][k])[:100] 



    def finalise_decoding(self):
        """ this incorporates the probability of terminating at each state
        """
        
        for state in self.f.states():
            final_weight = float(self.f.final(state))
            if self.V[-1][state] != self.NLL_ZERO:
                if final_weight == math.inf:
                    self.V[-1][state] = self.NLL_ZERO  # effectively says that we can't end in this state
                else:
                    self.V[-1][state] += final_weight
                    
        # get a list of all states where there was a path ending with non-zero probability
        finished = [x for x in self.V[-1] if x < self.NLL_ZERO]
        if not finished:  # if empty
            print("No path got to the end of the observations.")
        
        
    def decode(self):
        self.initialise_decoding()
        t = 1
        while t <= self.om.observation_length():
            self.forward_step(t)
            self.traverse_epsilon_arcs(t)
            t += 1
        self.finalise_decoding()
        return t-1

    
    def backtrace(self):
        
        best_final_state = self.V[-1].index(min(self.V[-1])) # argmin
        best_state_sequence = [best_final_state]
        best_out_sequence = []
        
        t = self.om.observation_length()   # ie T
        j = best_final_state
        
        while t >= 0:
            i = self.B[t][j]
            best_state_sequence.append(i)
            best_out_sequence = self.W[t][j] + best_out_sequence  # computer scientists might like
                                                                                # to make this more efficient!

            # continue the backtrace at state i, time t-1
            j = i  
            t-=1
            
        best_state_sequence.reverse()
        
        # convert the best output sequence from FST integer labels into strings
        best_out_sequence = ' '.join([ self.f.output_symbols().find(label) for label in best_out_sequence])
        
        return (best_state_sequence, best_out_sequence)

def parse_lexicon(lex_file):
    lex = {}  # create a dictionary for the lexicon entries (this could be a problem with larger lexica)
    with open(lex_file, 'r') as f:
        for line in f:
            line = line.split()  # split at each space
            lex[line[0]] = line[1:]  # first field the word, the rest is the phones
    return lex

def generate_symbol_tables(lexicon, n=3):
    state_table = fst.SymbolTable()
    phone_table = fst.SymbolTable()
    word_table = fst.SymbolTable()
    
    # add empty <eps> symbol to all tables
    state_table.add_symbol('<eps>')
    phone_table.add_symbol('<eps>')
    word_table.add_symbol('<eps>')
    state_table.add_symbol('<eps>')
    phone_table.add_symbol('sil')
    state_table.add_symbol('sil_1')
    state_table.add_symbol('sil_2')
    state_table.add_symbol('sil_3')
    state_table.add_symbol('sil_4')
    #state_table.add_symbol('sil_5')
    for word, phones  in lexicon.items():
        
        word_table.add_symbol(word)
        
        for p in phones: # for each phone
            
            phone_table.add_symbol(p)
            for i in range(1,n+1): # for each state 1 to n
                state_table.add_symbol('{}_{}'.format(p, i))
            
    return word_table, phone_table, state_table


def generate_word_wfst(f, start_state, word):
    """ Generate a WFST for any word in the lexicon, composed of 3-state phone WFSTs.
        This will currently output phone labels.  
    """ 
    current_state = start_state
    
    # iterate over all the phones in the word
    for phone in range(lex[word]):   # will raise an exception if word is not in the lexicon        
        current_state = generate_phone_wfst(f, current_state, phone)
        # note: new current_state is now set to the final state of the previous phone WFST
    f.set_final(current_state)
    
    return f


def generate_phone_wfst(f, start_state, phone, n):
    global count_states
    global count_arcs
    current_state = start_state
    for i in range(1, n+1):
        
        in_label = state_table.find('{}_{}'.format(phone, i))
        
        sl_weight = fst.Weight('log', -math.log(0.5))  # weight for self-loop
        # self-loop back to current state
        f.add_arc(current_state, fst.Arc(in_label, 0, sl_weight, current_state))
        count_arcs += 1

        if i == n:
            out_label = phone_table.find(phone)
        else:
            out_label = 0   # output empty <eps> label
            
        next_state = f.add_state()
        count_states += 1
        next_weight = fst.Weight('log', -math.log(0.5)) # weight to next state
        f.add_arc(current_state, fst.Arc(in_label, out_label, next_weight, next_state))    
        count_arcs += 1
        current_state = next_state
        
    return current_state

def generate_silence(f, start_state, phone, n, end_state):
    global count_states
    global count_arcs
    current_state = start_state
    for i in range(1, n+1):
        
        in_label = state_table.find('{}_{}'.format(phone, i))
        
        sl_weight = fst.Weight('log', -math.log(0.7))  # weight for self-loop
        # self-loop back to current state
        f.add_arc(current_state, fst.Arc(in_label, 0, sl_weight, current_state))
        count_arcs += 1
        if i == n:
            next_state = end_state
        else:
            next_state = f.add_state()
            count_states += 1
        next_weight = fst.Weight('log', -math.log(0.3)) # weight to next state
        f.add_arc(current_state, fst.Arc(in_label, 0, next_weight, next_state))    
        count_arcs += 1
        current_state = next_state
        
    return current_state

def generate_word_sequence_recognition_wfst(n, lex):
    """ generate a HMM to recognise any single word sequence for words in the lexicon
    """
    global count_states
    global count_arcs
    f = fst.Fst('log')
    
    # create a single start state
    start_state = f.add_state()
    count_states += 1
    f.set_start(start_state)
    
    for word, phones in lex.items():
        current_state = f.add_state()
        count_states += 1
        arc_weight = fst.Weight('log', -math.log(1/len(lex)))
        f.add_arc(start_state, fst.Arc(0, 0, arc_weight, current_state))
        count_arcs += 1
        for phone in phones: 
            current_state = generate_phone_wfst(f, current_state, phone, n)
        f.set_final(current_state)
        
        current_state = generate_silence(f, current_state, 'sil', n+1, start_state)
        
        #arc_weight = fst.Weight('log', -math.log(0.5))
        #f.add_arc(current_state, fst.Arc(0, 0, arc_weight, start_state))
        #count_arcs += 1
    return f

def read_transcription(wav_file):
    """
    Get the transcription corresponding to wav_file.
    """
    
    transcription_file = os.path.splitext(wav_file)[0] + '.txt'
    
    with open(transcription_file, 'r') as f:
        transcription = f.readline().strip()
    
    return transcription

def phones_to_words(s, dic, result):
        if not s:          
            return ' '.join([dic[x] for x in result])
        j = 0
        s = s.strip()
        while j < len(s):
            if s[:j+1] in dic:
                result.append(s[:j+1])
                ans = phones_to_words(s[j+1:], dic, result)
                if ans:
                    return ans
                result.pop()
            j+=1

word_table, phone_table, state_table = None, None, None
count_states, count_arcs = 0, 0

if __name__ == "__main__":
    print(datetime.now().strftime("%H:%M:%S"))
    lex = parse_lexicon('lexicon.txt')
    phone_to_word = {' '.join(lex[x]):x for x in lex}
    word_table, phone_table, state_table = generate_symbol_tables(lex)
    f = generate_word_sequence_recognition_wfst(3, lex)
    f.set_input_symbols(state_table)
    f.set_output_symbols(phone_table)
    # from subprocess import check_call
    # from IPython.display import Image
    # f.draw('tmp.dot', portrait=True)
    # check_call(['dot','-Tpng','-Gdpi=500','tmp.dot','-o','tmp.png'])
    # Image(filename='tmp.png')

    WER, decode_time, backtrace_time, utterances, forward_ops = 0, 0, 0, 0, 0
    for wav_file in glob.glob('/group/teaching/asr/labs/recordings/000?.wav'):    # replace path if using your own                                                                         # audio files
        utterances += 1
        decoder = MyViterbiDecoder(f, wav_file)
        words = []
        
        start_time = timer()
        forward_ops += decoder.decode()
        end_time = timer()
        decode_time +=  end_time - start_time
        
        start_time = timer()
        (state_path, phones) = decoder.backtrace() 
        words = phones_to_words(phones, phone_to_word, words)
        end_time = timer()
        backtrace_time += end_time - start_time

        transcription = read_transcription(wav_file)
        error_counts = wer.compute_alignment_errors(transcription, words)
        word_count = len(transcription.split())
        WER += sum(error_counts)/word_count
    print(datetime.now().strftime("%H:%M:%S"))
    WER = WER/utterances
    backtrace_time = backtrace_time/utterances
    decode_time = decode_time/utterances
    forward_ops = forward_ops/utterances
    #count_states, count_arcs - Memory stats
    #WER - Accuracy stats
    #backtrace_time, decode_time, forward_ops (avg) - Speed stats
    print(WER, count_states, count_arcs, backtrace_time, decode_time, forward_ops)


