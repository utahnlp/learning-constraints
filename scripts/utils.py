import torch
import torch.nn as nn

def argmax(vec):
    # return the argmax as a python int
    _, idx = torch.max(vec, 1)
    return idx.item()


def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] for w in seq]
    return torch.tensor(idxs, dtype=torch.long)


# Compute log sum exp in a numerically stable way for the forward algorithm
def log_sum_exp(vec):
    max_score = vec[0, argmax(vec)]
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
    return max_score + \
        torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))

def prepare_word2idx(training_data):
	word_to_ix = {}
	for sentence, tags in training_data:
		for word in sentence:
			if word not in word_to_ix:
				word_to_ix[word] = len(word_to_ix)
	return word_to_ix

def prepare_tag2idx(training_data):
	tag_to_ix = {}
	for sentence, tags in training_data:
		for tag in tags:
			if tag not in tag_to_ix:
				tag_to_ix[tag] = len(tag_to_ix)
	return tag_to_ix


def calc_correct(pred,tag):
	predictions = pred[1]

	c = 0
	t = 0
	for i in range(len(tag)):
		if tag[i] == predictions[i]:
			c+=1
		t+=1

	return c,t