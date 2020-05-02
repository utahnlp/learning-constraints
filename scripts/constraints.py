from utils import prepare_sequence
import torch
import random
import time
#import nltk
import spacy
import itertools
#from nltk.data import load


nlp = spacy.load("en_core_web_sm")

random.seed(2)
state = random.getstate() 
random.setstate(state)
start_emb = [random.random() for _ in range(300)]
stop_emb = [random.random() for _ in range(300)]
null_emb = [random.random() for _ in range(300)]
opt_name = ['Ngram_Label','POS-tag']


##############################################################################################
# 								N-GRAM LABEL LOCAL CONSTRAINT 							 	 #
##############################################################################################
def get_gram_vector(choice,gram,tag_to_ix,data,embedding_matrix, sentence,tokens_sent, pos_tags):
	
	if data=="dummy":
		return torch.tensor([0]*(gram*len(tag_to_ix)))
	else:
		doi = data[-gram:]
		temp_data = [None]*(gram-len(doi))
		temp_data.extend(doi)

		temp_arr = [0]*(gram*len(tag_to_ix))
		for i in range(len(temp_data)):
			if temp_data[i] == None:
				continue
			else:
				temp_arr[i*len(tag_to_ix) + temp_data[i]] = 1

		return torch.tensor([temp_arr])


def get_positive_grams(targets,gram,tag_to_ix):
	gram_set = set()	
	if gram > 1:
		new_targets = [None]*(gram-2)
		new_targets.append(tag_to_ix["<START>"])
		new_targets.extend(targets)
		new_targets.append(tag_to_ix["<STOP>"])
		new_targets.extend([None]*(gram-2))
		# new_targets = [tag_to_ix["<START>"]]*(gram-1)
		# new_targets.extend(targets)
		# new_targets.extend([tag_to_ix["<STOP>"]]*(gram-1))
	else:
		new_targets = targets

	for i in range(len(new_targets)-gram+1):
		temp_arr = [0]*(gram*len(tag_to_ix))
		for j in range(gram):
			if new_targets[i+j]==None:
				continue
			temp_arr[j*len(tag_to_ix) + new_targets[i+j]] = 1
		gram_set.add(tuple(temp_arr))
		
	return gram_set


def get_negative_grams(positive_ex,nminus1set,tag_to_ix,no_of_gram,train_pos):
	negative_ex = set()
	for arr in positive_ex:
		temp_arr = list(arr).copy()
		g_pointer = [*range(no_of_gram)]
		flag = 0
		while len(g_pointer)!=0:
			index = random.choice(g_pointer)
			g_pointer = set(g_pointer)
			g_pointer.remove(index)
			g_pointer = list(g_pointer)
			
			t_pointer = [*range(len(tag_to_ix))]
			while len(t_pointer) != 0:
				index_t = random.choice(t_pointer)
				t_pointer = set(t_pointer)
				t_pointer.remove(index_t)
				t_pointer = list(t_pointer)
				
				new = [0]*len(tag_to_ix)
				new[index_t] = 1
				new_arr = temp_arr.copy()
				new_arr[index*len(tag_to_ix):(index+1)*len(tag_to_ix)] = new
				
				if tuple(new_arr) in negative_ex:
					continue
				if tuple(new_arr) in positive_ex:
					continue
				if tuple(new_arr) in train_pos:
					continue
				else:
					if (tuple(new_arr[-(no_of_gram-1)*len(tag_to_ix):]) in nminus1set)  \
						and (tuple(new_arr[:(no_of_gram-1)*len(tag_to_ix)]) in nminus1set):
						negative_ex.add(tuple(new_arr))
						flag=1
						break
		
			if flag==1:
				break
			
	return negative_ex



def ngram_lab(data, grams, word_to_ix, tag_to_ix, embedding_matrix,train_pos,nminus1):
	positive_ex = set()
	nminus1set = set()	
	for sentence, tags in data:
		targets = [tag_to_ix[t] for t in tags]

		if grams  > 1:  
			nminus1set_temp = get_positive_grams(targets,grams-1,tag_to_ix)
			nminus1set = nminus1set.union(nminus1set_temp)
		n_set = get_positive_grams(targets,grams,tag_to_ix)
		positive_ex = positive_ex.union(n_set)

	negative_ex = get_negative_grams(positive_ex,nminus1set.union(nminus1),tag_to_ix,grams,train_pos)

	return positive_ex, negative_ex, nminus1set


##############################################################################################
# 								POS TAGS LOCAL CONSTRAINT 		 							 #
##############################################################################################
def get_postag_vector(choice,gram,tag_to_ix,data,embedding_matrix,sentence,tokens_sent, pos_tags):
	tagdict = ['ADJ','ADP','ADV','AUX','CCONJ','DET','INTJ',
				'NOUN','NUM','PART','PRON','PROPN','PUNCT',
				'SCONJ','SYM','VERB','X'] 
	cnt = 0
	pos_tagdict = {}
	for key in tagdict:
		pos_tagdict[key] = cnt
		cnt+=1

	pos_tagdict["start"] = len(pos_tagdict)
	pos_tagdict["stop"] = len(pos_tagdict)

	if data=="dummy":
		return torch.tensor([0]*(gram*(len(pos_tagdict) + len(tag_to_ix))))
	else:
		pos_tags = pos_tags[:len(tokens_sent)]
		pos = []
		
		for t in pos_tags:
			pos.append(pos_tagdict[t])

		doi = data[-gram:]
		poi = pos[-gram:]

		temp_data = [None]*(gram-len(doi))
		temp_data.extend(doi)

		temp_pos = [None]*(gram -len(poi))
		temp_pos.extend(poi)
		
		if len(poi) < gram:
			temp_pos[gram-len(poi)-1] =  pos_tagdict["start"]


		temp_arr = [0]*(gram*len(tag_to_ix))
		temp_pos_arr = [0]*(gram*len(pos_tagdict))

		for j in range(gram):
			if temp_data[j]==None:
				continue
			temp_arr[j*len(tag_to_ix) + temp_data[j]] = 1
			temp_pos_arr[j*len(pos_tagdict) + temp_pos[j]] = 1

		temp_arr.extend(temp_pos_arr)

		return torch.tensor([temp_arr])

def get_negative_pos_tag(positive_ex,tag_to_ix,no_of_gram, train_pos):
	negative_set = set()
	tag_index = len(list(positive_ex)[0])-no_of_gram*len(tag_to_ix)

	for arr in positive_ex:
		temp_arr = list(arr).copy()
		g_pointer = [*range(no_of_gram)] 
		flag = 0
		while len(g_pointer)!=0:
			index = random.choice(g_pointer)
			g_pointer = set(g_pointer)
			g_pointer.remove(index)
			g_pointer = list(g_pointer)
			
			t_pointer = [*range(len(tag_to_ix))]
			while len(t_pointer) != 0:
				index_t = random.choice(t_pointer)
				t_pointer = set(t_pointer)
				t_pointer.remove(index_t)
				t_pointer = list(t_pointer)
				
				new = [0]*len(tag_to_ix)
				new[index_t] = 1
				new_arr = temp_arr.copy()
				new_arr[tag_index + index*len(tag_to_ix): tag_index +(index+1)*len(tag_to_ix)] = new

				if tuple(new_arr) in negative_set:
					continue
				if tuple(new_arr) in positive_ex:
					continue
				if tuple(new_arr) in train_pos:
					continue
				else:
					negative_set.add(tuple(new_arr))
					flag=1
					break
		
			if flag==1:
				break

	return negative_set


def get_positive_pos(pos,targets,gram,tag_to_ix,pos_tagdict):
	gram_set = set()

	if gram > 1:
		new_targets = [None]*(gram-2)
		new_pos = [None]*(gram-2)
		new_targets.append(tag_to_ix["<START>"])
		new_pos.append(pos_tagdict["start"])
		new_targets.extend(targets)
		new_pos.extend(pos)
		new_targets.append(tag_to_ix["<STOP>"])
		new_pos.append(pos_tagdict["stop"])
		new_targets.extend([None]*(gram-2))
		new_pos.extend([None]*(gram-2))
	else:
		new_targets = targets
		new_pos = pos

	for i in range(len(new_targets)-gram+1):
		temp_arr = [0]*(gram*len(tag_to_ix))
		temp_pos_arr = [0]*(gram*len(pos_tagdict))

		for j in range(gram):
			if new_targets[i+j]==None:
				continue
			temp_arr[j*len(tag_to_ix) + new_targets[i+j]] = 1
			temp_pos_arr[j*len(pos_tagdict) + new_pos[i+j]] = 1

		temp_arr.extend(temp_pos_arr)
		gram_set.add(tuple(temp_arr))

	return gram_set


def pos_tag(data, grams, word_to_ix, tag_to_ix, embedding_matrix,train_pos,nminus1):
	positive_ex = set()
	tagdict = ['ADJ','ADP','ADV','AUX','CCONJ','DET','INTJ',
				'NOUN','NUM','PART','PRON','PROPN','PUNCT',
				'SCONJ','SYM','VERB','X'] 
	cnt = 0
	pos_tagdict = {}
	for key in tagdict:
		pos_tagdict[key] = cnt
		cnt+=1

	pos_tagdict["start"] = len(pos_tagdict)
	pos_tagdict["stop"] = len(pos_tagdict)
	
	for sentence, tags in data:		
		targets = [tag_to_ix[t] for t in tags]

		pos = []
		tokens = nlp.tokenizer.tokens_from_list(sentence)
		nlp.tagger(tokens)
		for t in tokens:
			pos.append(pos_tagdict[t.pos_])
							
		n_set = get_positive_pos(pos,targets,grams,tag_to_ix,pos_tagdict)

		positive_ex = positive_ex.union(n_set)
	
	negative_ex = get_negative_pos_tag(positive_ex,tag_to_ix,grams,train_pos)
	
	return positive_ex, negative_ex, nminus1



func_dict = [ngram_lab, pos_tag]
vec_dict = [get_gram_vector,get_postag_vector]

def get_constraint_features(data,opt,grams,word_to_ix, tag_to_ix, embedding_matrix, train_pos=set(),nminus1=set()):
	print("Preparing constraint features for {} considering {} grams".format(opt_name[opt-1],grams))
	positive_ex, negative_ex, m1 = func_dict[opt-1](data, grams, word_to_ix, tag_to_ix, embedding_matrix,train_pos,nminus1)	

	return positive_ex, negative_ex, m1


def get_constraint_vector(choice, gram, tag_to_ix,data="dummy", embedding_matrix = None, sentence=None, tokens_sent=None, pos = None):	
	vec = vec_dict[choice-1](choice,gram,tag_to_ix,data,embedding_matrix,sentence,tokens_sent,pos)
	return vec