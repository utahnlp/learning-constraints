import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim
import torch.nn as nn
import argparse, os, json
from rectifier_net import RectifierNetwork
from preprocess import preprocess , prepare_word2idx_glove
from utils import prepare_word2idx, prepare_tag2idx
from constraints import get_constraint_features
import pickle

torch.manual_seed(2)
data_dir = "./../data/processed_files/"
#save_dir = "./../constraint_models_2/"

def config_parser(parser):
	""" Add all command line arguments here
	"""
	parser.add_argument('--choice', type=int, choices=range(1, 3), default = 1,
						help="Your constraint choices- \n 1) Label N-gram existence \n2) Part of Speech Tags")
	parser.add_argument('--ngram', type=int, default=1)
	parser.add_argument('--train_dir', type=str, default="./../data/train_new.txt")
	parser.add_argument('--test_dir', type=str, default="./../data/test_new.txt")
	parser.add_argument('--dev_dir', type=str, default="./../data/dev_new.txt")
	parser.add_argument('--save_dir', type=str, default="./../constraint_models_2/")
	parser.add_argument('--ratio', type=float, default=1.0)
	parser.add_argument('--hidden_rect', type=int, default=10)
	parser.add_argument('--task_name',type=str, default="chunking")
	parser.add_argument('--save_enable',type=int, default=0)
	parser.add_argument('--lr',type=float, default=0.001)
	
	return parser

def convertSettoList(pstv_ex, neg_ex):
	pstv_list = list(pstv_ex)
	#pstv = [list(elem) for elem in pstv_list]
	pstv = list(map(list, pstv_list))

	neg_list = list(neg_ex)
	#neg = [list(elem) for elem in neg_list]
	neg = list(map(list, neg_list))

	data = pstv.copy()
	lab = [1]*len(pstv)
	lab.extend([0]*len(neg))
	data.extend(neg)

	return data, lab



def getVectors(args):
	""" Based on the training, dev and test files, we get the positive and negative
	vectors with their respective labels for the rectifier training 
	"""
	START_TAG = "<START>"
	STOP_TAG = "<STOP>"

	if os.path.exists(data_dir+args['task_name']+"_"+str(args['ratio'])+".pkl"):
		print("preprocessed dumps found. Loading them.")
		with open(data_dir+args['task_name']+"_"+str(args['ratio'])+".pkl", 'rb') as handle:
			b = pickle.load(handle)

		word_to_ix = b['word_to_ix']
		training_data = b['training_data']
		dev_data = b['dev_data']
		test_data = b['test_data']
		embedding_matrix = b['embedding_matrix']
	else:
		print("Saved preprocessed data files not found .....creating new dump")
		training_data,dev_data,test_data = preprocess(args['ratio'],
														args['train_dir'],
														args['dev_dir'],
														args['test_dir'])

		
		word_to_ix, training_data, dev_data, test_data, embedding_matrix = prepare_word2idx_glove(training_data,dev_data, test_data)

		save_dict = {"word_to_ix":word_to_ix, "training_data":training_data, "dev_data":dev_data, "test_data":test_data, "embedding_matrix":embedding_matrix}

		with open(data_dir+args['task_name']+"_"+str(args['ratio'])+".pkl", 'wb') as handle:
			pickle.dump(save_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

		print("New preprocessed files dumped.....Things'll get faster next time!")


	data = training_data.copy()
	data.extend(dev_data)
	data.extend(test_data)

	#word_to_ix = prepare_word2idx(data)
	
	tag_to_ix  = prepare_tag2idx(data)
	tag_to_ix[START_TAG] = len(tag_to_ix.keys())
	tag_to_ix[STOP_TAG] = len(tag_to_ix.keys())


	pstv_ex, neg_ex, nminus1 = get_constraint_features(training_data, 
												args['choice'],
												args['ngram'],
												word_to_ix,
												tag_to_ix,
												embedding_matrix)

	pstv_dev_ex, neg_dev_ex, _ = get_constraint_features(dev_data, 
													args['choice'],
													args['ngram'],
													word_to_ix,
													tag_to_ix,
													embedding_matrix,
													train_pos = pstv_ex,
													nminus1 = nminus1)
	print("Positive Examples: " +str(len(pstv_ex)))
	print("Negative Examples: " +str(len(neg_ex)))
	print("Dev Positive Examples: " +str(len(pstv_dev_ex)))
	print("Dev Negative Examples: " +str(len(neg_dev_ex)))

	assert len(pstv_ex.intersection(neg_ex)) == 0
	assert len(pstv_ex.intersection(neg_dev_ex)) == 0
	assert len(pstv_dev_ex.intersection(neg_dev_ex)) == 0

	data, lab = convertSettoList(pstv_ex, neg_ex)
	data_dev, lab_dev = convertSettoList(pstv_dev_ex, neg_dev_ex) 

	return data,lab, data_dev, lab_dev



def predict(net, loader):
	correct = 0
	total=0
	for index, (inp, gold) in enumerate(loader):
		pred = net(inp)

		for i in range(len(pred)):
			if pred[i] >= 0.5:
				pred_b = 1
			else:
				pred_b = 0

			if pred_b == gold[i]: 
				correct += 1			
			total+=1
		
	return correct/total 

	

def train(data, lab, data_dev, lab_dev):
	""" Function to train the rectifier network
	"""

	# Data Loaders are formed for the data splits
	dataset = TensorDataset(torch.tensor(data).float(), torch.tensor(lab).float())
	loader = DataLoader(dataset, batch_size=8,shuffle=True)

	dataset_dev = TensorDataset(torch.tensor(data_dev).float(), torch.tensor(lab_dev).float())
	loader_dev = DataLoader(dataset_dev, batch_size=8,shuffle=True)
	
	net = RectifierNetwork(len(data[0]),args['hidden_rect'])
	optimizer = optim.Adam(net.parameters(),lr = args['lr'])
	loss = nn.BCELoss()

	if not os.path.exists(save_dir+"choice_"+str(args['choice'])+"/"):
		os.mkdir(save_dir+"choice_"+str(args['choice'])+"/")

	if not os.path.exists(save_dir+"choice_"+str(args['choice'])+"/ngram_"+str(args['ngram'])+"/"):
		os.mkdir(save_dir+"choice_"+str(args['choice'])+"/ngram_"+str(args['ngram'])+"/")

	if not os.path.exists(save_dir+"choice_"+str(args['choice'])+"/ngram_"+str(args['ngram'])+"/ratio_"+str(args['ratio'])+"/"):
		os.mkdir(save_dir+"choice_"+str(args['choice'])+"/ngram_"+str(args['ngram'])+"/ratio_"+str(args['ratio'])+"/")

	save_folder = save_dir+"choice_"+str(args['choice'])+"/ngram_"+str(args['ngram'])+"/ratio_"+str(args['ratio'])+"/"

	prev_best_dev = 0.0
	for ep in range(1000):
		# Training Loop
		train_loss=0.0
		for index, (inp, gold) in enumerate(loader):
			optimizer.zero_grad()
			output = net(inp)
			out_loss = loss(output,gold) 
			train_loss+= out_loss.item()
			out_loss.backward()
			optimizer.step()
			
		train_acc	= predict(net,loader)
		dev_acc		= predict(net, loader_dev) 	

		# Models with better dev accuracy are saved	
		if dev_acc> prev_best_dev:
			prev_best_dev = dev_acc
			if args['save_enable']:
				torch.save({    'epoch': ep+1,
					            'model_state_dict': net.state_dict(),
					            'optimizer_state_dict': optimizer.state_dict(),
					            'loss': train_loss/len(data),
					            'dev_accuracy': dev_acc,
					            'train_accuracy': train_acc
					        }, save_folder+str(ep+1)+"_"+str(dev_acc))
				
				res = {'dev_acc': dev_acc, 'train_acc':train_acc}

				with open(save_folder+str(ep+1)+"_"+str(dev_acc)+".json", 'w') as fp:
					json.dump(res, fp)


		print("Epoch : {}".format(ep+1))
		print("Training loss: {:.4f}".format(train_loss/len(data)))	
		print("Training Accuracy: {:.4f} ".format(train_acc))	
		print("Development Accuracy: {:.4f} \n".format(dev_acc))	

		if train_acc == 1.0:
			print("Train Accuracy reached 1. Ending early!!")
			exit()



if __name__=="__main__":
	parser = argparse.ArgumentParser()
	parser = config_parser(parser)
	args = vars(parser.parse_args())
	save_dir = args["save_dir"]
	data, lab, data_dev, lab_dev = getVectors(args)
	train(data,lab, data_dev, lab_dev)
	
	

