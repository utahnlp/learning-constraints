import numpy as np
import time

def preprocess(ratio=1, 
				train_file = "./../data/train_new.txt",
				dev_file = "./../data/dev_new.txt",
				test_file = "./../data/test_new.txt"):

	train_data = open(train_file,"r").readlines()
	dev_data = open(dev_file,"r").readlines()
	test_data = open(test_file,"r").readlines()

	train_d = []
	dev_d = []
	test_d = []

	temp_wrd = []
	temp_lab = []

	for line in train_data:
		if line =="\n":
			train_d.append((temp_wrd,temp_lab))
			temp_wrd = []
			temp_lab = []
			continue
		lab = line.split(" ")[1].strip("\n")
		word = line.split(" ")[0]
		temp_wrd.append(word)
		temp_lab.append(lab)

	
	temp_wrd = []
	temp_lab = []

	for line in dev_data:
		if line =="\n":
			dev_d.append((temp_wrd,temp_lab))
			temp_wrd = []
			temp_lab = []
			continue
		lab = line.split(" ")[1].strip("\n")
		word = line.split(" ")[0]
		temp_wrd.append(word)
		temp_lab.append(lab)

	train_d = train_d[:int(len(train_d)*ratio)]
	dev_d = dev_d[:int(len(dev_d)*ratio)]
	
	temp_wrd = []
	temp_lab = []

	for line in test_data:
		if line =="\n":
			test_d.append((temp_wrd,temp_lab))
			temp_wrd = []
			temp_lab = []
			continue
		lab = line.split(" ")[1].strip("\n")
		word = line.split(" ")[0]
		temp_wrd.append(word)
		temp_lab.append(lab)

	print(len(train_d))
	print(len(dev_d))
	print(len(test_d))

	return train_d,dev_d,test_d


def loadGloveModel(gloveFile):
    print("Loading Glove Vocab")
    f = open(gloveFile,'r')
    #f2 = open(gloveFile,'r').readlines()
    model = {}
    for line in f:
        splitLine = line.split()
        word = splitLine[0]
        try:
        	embedding = np.array([float(val) for val in splitLine[1:]])
        	model[word] = embedding
        except:
        	continue
    print("Done.",len(model)," words loaded!")
    return model


def prepare_word2idx_glove(training_data, dev_data, test_data):
	word_to_ix = {'unk':0}
	embed = loadGloveModel("./../data/glove.840B.300d.txt")
	glove_vocab = embed.keys()
	embedding_matrix = [embed['unk']]
	
	for i in range(0,len(training_data)):
		for j in range(0,len(training_data[i][0])):
			if training_data[i][0][j] not in word_to_ix.keys():
				if training_data[i][0][j] not in glove_vocab:
					training_data[i][0][j] = 'unk'
				else:
					word_to_ix[training_data[i][0][j]] = len(word_to_ix)
					embedding_matrix.append(embed[training_data[i][0][j]])
		
	for i in range(0,len(dev_data)):
		for j in range(0,len(dev_data[i][0])):
			if dev_data[i][0][j] not in word_to_ix.keys():
				if dev_data[i][0][j] not in glove_vocab:
					dev_data[i][0][j] = 'unk'
				else:
					word_to_ix[dev_data[i][0][j]] = len(word_to_ix)
					embedding_matrix.append(embed[dev_data[i][0][j]])
	
	for i in range(0,len(test_data)):
		for j in range(0,len(test_data[i][0])):
			if test_data[i][0][j] not in word_to_ix.keys():
				if test_data[i][0][j] not in glove_vocab:
					test_data[i][0][j] = 'unk'
				else:
					word_to_ix[test_data[i][0][j]] = len(word_to_ix)
					embedding_matrix.append(embed[test_data[i][0][j]])
	

	print("Loaded GloVe vectors")
	del embed
	
	return word_to_ix, training_data, dev_data, test_data, embedding_matrix