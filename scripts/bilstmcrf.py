import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
from utils import argmax, log_sum_exp
from rectifier_net import RectifierNetwork
from constraints import get_constraint_vector
import time

START_TAG = "<START>"
STOP_TAG = "<STOP>"


class BiLSTM_CRF(nn.Module):

    def __init__(self, vocab_size, tag_to_ix, embedding_dim, hidden_dim, embed_matrix=None):
        super(BiLSTM_CRF, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.tag_to_ix = tag_to_ix
        self.tagset_size = len(tag_to_ix)

        if embed_matrix==None:
            self.word_embeds = nn.Embedding(vocab_size, embedding_dim)
        else:
            self.word_embeds = nn.Embedding.from_pretrained(torch.tensor(embed_matrix).float())

        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2,
                            num_layers=1, bidirectional=True)

        # Maps the output of the LSTM into tag space.
        self.hidden2tag = nn.Linear(hidden_dim, self.tagset_size)

        # Matrix of transition parameters.  Entry i,j is the score of
        # transitioning *to* i *from* j.
        self.transitions = nn.Parameter(
            torch.randn(self.tagset_size, self.tagset_size))

        # These two statements enforce the constraint that we never transfer
        # to the start tag and we never transfer from the stop tag
        self.transitions.data[tag_to_ix[START_TAG], :] = -10000
        self.transitions.data[:, tag_to_ix[STOP_TAG]] = -10000

        self.hidden = self.init_hidden()

    def init_hidden(self):
        return (torch.zeros(2, 1, self.hidden_dim // 2),
                torch.zeros(2, 1, self.hidden_dim // 2))

    def _forward_alg(self, feats):
        # Do the forward algorithm to compute the partition function
        init_alphas = torch.full((1, self.tagset_size), -10000.)
        # START_TAG has all of the score.
        init_alphas[0][self.tag_to_ix[START_TAG]] = 0.
        
        # Wrap in a variable so that we will get automatic backprop
        forward_var = init_alphas

        # Iterate through the sentence
        for feat in feats:
            alphas_t = []  # The forward tensors at this timestep
            for next_tag in range(self.tagset_size):
                # broadcast the emission score: it is the same regardless of
                # the previous tag
                emit_score = feat[next_tag].view(
                    1, -1).expand(1, self.tagset_size)
                # the ith entry of trans_score is the score of transitioning to
                # next_tag from i
                trans_score = self.transitions[next_tag].view(1, -1)

                # The ith entry of next_tag_var is the value for the
                # edge (i -> next_tag) before we do log-sum-exp
                next_tag_var = forward_var + trans_score + emit_score
                # The forward variable for this tag is log-sum-exp of all the
                # scores.
               
                alphas_t.append(log_sum_exp(next_tag_var).view(1))
            
            forward_var = torch.cat(alphas_t).view(1, -1)

        terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]
        alpha = log_sum_exp(terminal_var)

        return alpha

    def _get_lstm_features(self, sentence):
        self.hidden = self.init_hidden()
        embeds = self.word_embeds(sentence).view(len(sentence), 1, -1)
        lstm_out, self.hidden = self.lstm(embeds, self.hidden)
        lstm_out = lstm_out.view(len(sentence), self.hidden_dim)
        lstm_feats = self.hidden2tag(lstm_out)
        
        return lstm_feats

    def _score_sentence(self, feats, tags):
        # Gives the score of a provided tag sequence
        score = torch.zeros(1)
        tags = torch.cat([torch.tensor([self.tag_to_ix[START_TAG]], dtype=torch.long), tags])
        for i, feat in enumerate(feats):
            score = score + \
                self.transitions[tags[i + 1], tags[i]] + feat[tags[i + 1]]
        score = score + self.transitions[self.tag_to_ix[STOP_TAG], tags[-1]]
        return score

    def _viterbi_decode(self, feats):
        backpointers = []
        forwards = []

        # Initialize the viterbi variables in log space
        init_vvars = torch.full((1, self.tagset_size), -10000.)

        init_vvars[0][self.tag_to_ix[START_TAG]] = 0

        # forward_var at step i holds the viterbi variables for step i-1
        forward_var = init_vvars
        for feat in feats:
            bptrs_t = []  # holds the backpointers for this step
            viterbivars_t = []  # holds the viterbi variables for this step

            for next_tag in range(self.tagset_size):
                # next_tag_var[i] holds the viterbi variable for tag i at the
                # previous step, plus the score of transitioning
                # from tag i to next_tag.
                # We don't include the emission scores here because the max
                # does not depend on them (we add them in below)
                next_tag_var = forward_var + self.transitions[next_tag]
                best_tag_id = argmax(next_tag_var)
                bptrs_t.append(best_tag_id)
                viterbivars_t.append(next_tag_var[0][best_tag_id].view(1))
            # Now add in the emission scores, and assign forward_var to the set
            # of viterbi variables we just computed
            forward_var = (torch.cat(viterbivars_t) + feat).view(1, -1)
            forwards.append(forward_var)
            backpointers.append(bptrs_t)
       
        # Transition to STOP_TAG
        terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]
        best_tag_id = argmax(terminal_var)
        path_score = terminal_var[0][best_tag_id]
        
        # Follow the back pointers to decode the best path.
        best_path = [best_tag_id]
        for bptrs_t in reversed(backpointers):
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append(best_tag_id)
        # Pop off the start tag (we dont want to return that to the caller)
        start = best_path.pop()
        assert start == self.tag_to_ix[START_TAG]  # Sanity check
        best_path.reverse()
        #print(best_path)

        return path_score, best_path

    def neg_log_likelihood(self, sentence, tags):
        feats = self._get_lstm_features(sentence)
        forward_score = self._forward_alg(feats)
        gold_score = self._score_sentence(feats, tags)
        return forward_score - gold_score

    def forward(self, sentence):  # dont confuse this with _forward_alg above.
        # Get the emission scores from the BiLSTM
        lstm_feats = self._get_lstm_features(sentence)

        # Find the best path, given the features.
        score, tag_seq = self._viterbi_decode(lstm_feats)
        return score, tag_seq

    def constrained_beam_search(self,sentence,lstm_feats,args,embedding_matrix,tokens,pos):
        
        log_softmax = nn.LogSoftmax(dim=0)
        beam_arr = [([self.tag_to_ix[START_TAG]],0.0)]

        if args['allow_constraint'] == 1:
            # Loading the trained rectifier network for constrained inference
            # Dummy vector returns a dummy vector of the inout shape of the rectifier
            dummy_vec = get_constraint_vector(args['choice'], args['ngram'], self.tag_to_ix,embedding_matrix=embedding_matrix)
            net = RectifierNetwork(len(dummy_vec),args['hidden_rect'])
            checkpoint = torch.load(args['constraint_model_dir'])
            net.load_state_dict(checkpoint['model_state_dict'])
            net.eval()
            
        itert=0
        for feat in lstm_feats:
            itert+=1
            temp_beam = []
            emission = log_softmax(feat).tolist()

            for beam in beam_arr:
                # For every beam compute the label log probabilities 
                # adding all possible beams for the step
                prev_tag = beam[0][-1]

                trans_scores = log_softmax(self.transitions[:,prev_tag]).tolist() 
                for i in range(self.tagset_size):
                    tag_seq = beam[0].copy()
                    temp_score = beam[1]
                    temp_score += emission[i] + trans_scores[i]
                    tag_seq.append(i)
                    temp_beam.append((tag_seq,temp_score)) 

            temp_beam.sort(key=lambda tup: tup[1],reverse=True)  # sorts in place

            
            new_beam = []
            
            ######### LOCAL CONSTRAINT CHECK ####################################################
            if args['allow_constraint']:
                i=0
                for index in range(len(temp_beam)):
                    # Gets corresponding constraint feature vector for every beam
                    vec = get_constraint_vector(args['choice'], args['ngram'], self.tag_to_ix, \
                                                temp_beam[index][0],self.word_embeds, \
                                                sentence[:len(temp_beam[index][0])-1], \
                                                tokens[:len(temp_beam[index][0])-1], pos)
                    satisfied = net(vec.float())        # Checks if constraint is satisfied

                    if satisfied[0] > 0.5:
                        i+=1
                        new_beam.append(temp_beam[index])
                    else:
                        temp_beam[index] = (temp_beam[index][0],-10000)
                      
                    # Break when you have the required amount of beams
                    if i == args['beam']:
                        break
                    
                
                if len(new_beam) == 0:
                    beam_arr = [temp_beam[0]]
                else:
                    beam_arr = new_beam

                
                
            else:
                # If constraint not allowed or choice is global 
                # we should continue with the normal BS decoder
                beam_arr = temp_beam[:args['beam']]
            
        
        ######### GLOBAL CONSTRAINT CHECK ####################################################
        ########## Irrelevant to the current constraints however should be adjusted ##########
        ####################### when global constraints are added ############################
        if args['choice'] > 2 and args['allow_constraint']:
            for index in range(len(beam_arr)):
                    vec = get_constraint_vector(args['choice'], args['ngram'], self.tag_to_ix, \
                                                beam_arr[index][0],embedding_matrix, sentence)
                    
                    satisfied = net(vec.float())

                    if satisfied[0] > 0.5:
                        return (beam_arr[index][1],beam_arr[index][0][1:])
                        
        return (beam_arr[0][1],beam_arr[0][0][1:])



    def predict(self,args,sentence,embedding_matrix,tokens,pos):
        # Get the emission scores from the BiLSTM
        lstm_feats = self._get_lstm_features(sentence)

        out = self.constrained_beam_search(sentence,lstm_feats,args,embedding_matrix,tokens,pos)
        
        return out




