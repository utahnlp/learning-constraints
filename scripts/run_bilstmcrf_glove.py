import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
from bilstmcrf  import BiLSTM_CRF
from utils import prepare_sequence, prepare_word2idx, calc_correct, prepare_tag2idx
import json, time, os, pickle
from tqdm import tqdm
from preprocess import preprocess , prepare_word2idx_glove
import argparse
import spacy

nlp = spacy.load("en_core_web_sm")
torch.manual_seed(1)

START_TAG = "<START>"
STOP_TAG = "<STOP>"

model_dir = "./../models_glove/"			# Directory where base models are saved
data_dir  = "./../data/processed_files/"	# Directory where processed data files are stored

toy = False
EMBEDDING_DIM = 300
HIDDEN_DIM = 128

def config_parser(parser):
	""" Add all command line arguments here
	"""
	parser.add_argument('--choice', type=int, choices=range(1, 3), default = 1,
						help="Your constraint choices- \n 1) Label N-gram existence \n2) Part of Speech Tags")
	parser.add_argument('--ngram', type=int, default=1)
	parser.add_argument('--train_dir', type=str, default="./../data/train_new.txt")
	parser.add_argument('--test_dir', type=str, default="./../data/test_new.txt")
	parser.add_argument('--dev_dir', type=str, default="./../data/dev_new.txt")
	parser.add_argument('--ratio', type=float, default=1.0)
	parser.add_argument('--mode', type=str, default="train")
	parser.add_argument('--model_dir', type=str, default="./../models_glove/0.01/16_0.8589211618257261")
	parser.add_argument('--constraint_model_dir', type=str, default="./../constraint_models/choice_1/ngram_3/ratio_0.01/90_0.8839285714285714")
	parser.add_argument('--save_enable', type=str, default="n")
	parser.add_argument('--task_name',type=str, default="chunking")
	parser.add_argument('--beam',type=int, default=5)
	parser.add_argument('--hidden_rect', type=int, default=10)
	parser.add_argument('--allow_constraint', type=int, default=1)

	return parser


def get_vectors(args):
	if not toy:
	    training_data,dev_data,test_data = preprocess(args['ratio'],
														args['train_dir'],
														args['dev_dir'],
														args['test_dir'])
	    word_to_ix, training_data, dev_data, test_data, embedding_matrix = prepare_word2idx_glove(training_data,dev_data, test_data)
	    data = training_data.copy()
	    data.extend(dev_data)
	    data.extend(test_data)
	    tag_to_ix  = prepare_tag2idx(data)
	    tag_to_ix[START_TAG] = len(tag_to_ix.keys())
	    tag_to_ix[STOP_TAG] = len(tag_to_ix.keys())
	else:
		EMBEDDING_DIM = 5
		HIDDEN_DIM = 4
		# Make up some training data
		training_data = [(
		    "the wall street journal reported today that apple corporation made money".split(),
		    "B I I I O O O B I O O".split()
		), (
		    "georgia tech is a university in georgia".split(),
		    "B I O O O O B".split()
		)]

		word_to_ix = prepare_word2idx(training_data)
		tag_to_ix = {"B": 0, "I": 1, "O": 2, START_TAG: 3, STOP_TAG: 4}
		dev_data = None
		test_data = None

	return training_data, dev_data, test_data, embedding_matrix, word_to_ix, tag_to_ix



def train(training_data, dev_data, embedding_matrix, test_data, word_to_ix, tag_to_ix):
	""" Function to train the chunker
	"""
	model = BiLSTM_CRF(len(word_to_ix), tag_to_ix, EMBEDDING_DIM, HIDDEN_DIM, embedding_matrix)
	optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-4)

	# checkpoint = torch.load("./../models_glove/26_0.9544608399545971")
	# model.load_state_dict(checkpoint['model_state_dict'])
	# optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
	# torch.set_rng_state(checkpoint['tst_rnd_state'])


	if toy:
		# Check predictions before training
		with torch.no_grad():
		    precheck_sent = prepare_sequence(training_data[0][0], word_to_ix)
		    precheck_tags = torch.tensor([tag_to_ix[t] for t in training_data[0][1]], dtype=torch.long)
		    print(model(precheck_sent))
	best_dev = 0
	early_stopping = 0
	# Make sure prepare_sequence from earlier in the LSTM section is loaded
	for epoch in range(300):  # again, normally you would NOT do 300 epochs, it is toy data
	    epoch_loss = 0
	    start = time.time()
	    print("Epoch "+str(epoch+1))
	    for sentence, tags in tqdm(training_data):
	        # Step 1. Remember that Pytorch accumulates gradients.
	        # We need to clear them out before each instance
	        model.zero_grad()

	        # Step 2. Get our inputs ready for the network, that is,
	        # turn them into Tensors of word indices.
	        sentence_in = prepare_sequence(sentence, word_to_ix)
	        targets = torch.tensor([tag_to_ix[t] for t in tags], dtype=torch.long)

	        # Step 3. Run our forward pass.
	        loss = model.neg_log_likelihood(sentence_in, targets)
	        epoch_loss += loss.item()
	        # Step 4. Compute the loss, gradients, and update the parameters by
	        # calling optimizer.step()
	        loss.backward()
	        optimizer.step()

	    end = time.time()
	    print("Loss : {}".format(epoch_loss/len(training_data)))
	    print("Time : {}".format(end-start))

	    dev_rnd_state = torch.get_rng_state()

		# Check predictions after training
	    with torch.no_grad():
	    	correct = 0
	    	total = 0
	    	for sentence, tags in dev_data:
	    		precheck_sent = prepare_sequence(sentence, word_to_ix)
	    		targets = [tag_to_ix[t] for t in tags]
	    		
	    		out = model(precheck_sent)
	    		corr, tot = calc_correct(out,targets)
	    		correct+= corr
	    		total += tot
	    	dev_acc = correct/total
	    	print("Dev accuracy (macro): {} \n".format(dev_acc))


	    tst_rnd_state = torch.get_rng_state()

		# Check predictions after training
	    with torch.no_grad():
	    	correct = 0
	    	total = 0
	    	for sentence, tags in test_data:
	    		precheck_sent = prepare_sequence(sentence, word_to_ix)
	    		targets = [tag_to_ix[t] for t in tags]

	    		out = model(precheck_sent)
	    		corr, tot = calc_correct(out,targets)
	    		
	    		correct+= corr
	    		total += tot
	    	test_acc = correct/total

	    # Save models
	    if args['save_enable'] == "y":
	        if not os.path.exists(model_dir+str(args["ratio"])+"/"):
	            os.mkdir(model_dir+str(args["ratio"])+"/")
	        torch.save(
		    	{
		            'epoch': epoch,
		            'model_state_dict': model.state_dict(),
		            'optimizer_state_dict': optimizer.state_dict(),
		            'loss': epoch_loss/len(training_data),
		            'dev_accuracy': dev_acc,
		            'test_accuracy': test_acc,
		            'dev_rnd_state': dev_rnd_state,
		            'tst_rnd_state': tst_rnd_state
		        }, model_dir+str(args["ratio"])+"/"+str(epoch+1)+"_"+str(dev_acc))

	        res = {'dev_acc': dev_acc, 'test_acc':test_acc}

	        with open(model_dir+str(args["ratio"])+"/"+str(epoch+1)+"_"+str(dev_acc)+".json", 'w') as fp:
	            json.dump(res, fp)


	    if dev_acc > best_dev:
	    	best_dev = dev_acc
	    	early_stopping = 0
	    else:
	    	early_stopping +=1
	    	if early_stopping == 10:
	    		exit()
	    


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser = config_parser(parser)
	args = vars(parser.parse_args())
	
	# Processed data files can be reused for all experiments
	# Checking if processed files exist and loading them, processing if not
	if os.path.exists(data_dir+args['task_name']+"_"+str(args['ratio'])+".pkl"):
		print("preprocessed dumps found. Loading them.")
		with open(data_dir+args['task_name']+"_"+str(args['ratio'])+".pkl", 'rb') as handle:
			b = pickle.load(handle)

		word_to_ix = b['word_to_ix']
		training_data = b['training_data']
		dev_data = b['dev_data']
		test_data = b['test_data']
		embedding_matrix = b['embedding_matrix']

		data = training_data.copy()
		data.extend(dev_data)
		data.extend(test_data)

		tag_to_ix  = prepare_tag2idx(data)
		tag_to_ix[START_TAG] = len(tag_to_ix.keys())
		tag_to_ix[STOP_TAG] = len(tag_to_ix.keys())

	else:
		print("Saved preprocessed data files not found .....creating new dump")
		if not os.path.exists(data_dir):
	            os.mkdir(data_dir)
		training_data, dev_data, test_data, embedding_matrix, word_to_ix, tag_to_ix = get_vectors(args)
		save_dict = {"word_to_ix":word_to_ix, "training_data":training_data, "dev_data":dev_data, "test_data":test_data, "embedding_matrix":embedding_matrix}

		with open(data_dir+args['task_name']+"_"+str(args['ratio'])+".pkl", 'wb') as handle:
			pickle.dump(save_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

		print("New preprocessed files dumped.....Things'll get faster next time!")

	
	if args['mode'] == "train":
		train(training_data, dev_data, embedding_matrix, test_data, word_to_ix, tag_to_ix)
	else:
		x = input("\nWe are proceeding with the following model file and ratio. Be absolutely sure that both the ratios are the same for correct results-\n"  \
			"Model File: {} \nRatio: {} \n(Press any 'y' to continue and 'n' to abort):".format(args['model_dir'],args['ratio']))
		if x =="n":
			exit()

		# Load the base model
		model = BiLSTM_CRF(len(word_to_ix), tag_to_ix, EMBEDDING_DIM, HIDDEN_DIM, embedding_matrix)
		checkpoint = torch.load(args['model_dir'])
		model.load_state_dict(checkpoint['model_state_dict'])
		model.eval()

		# Check predictions after training
		with torch.no_grad():
			correct = 0
			total = 0
			
			for sentence, tags in test_data:
				precheck_sent = prepare_sequence(sentence, word_to_ix)

				# Choice 2 has POS tags as constraint features
				# Hence loading POS tagger
				if args['choice']==2:									 
					pos = []
					tokens = nlp.tokenizer.tokens_from_list(sentence)
					nlp.tagger(tokens)
					for t in tokens:
						pos.append(t.pos_)
				else:
					pos = None

				targets = [tag_to_ix[t] for t in tags]
	
				out = model.predict(args,precheck_sent,embedding_matrix,sentence,pos)
				corr, tot = calc_correct(out,targets)

				correct+= corr
				total += tot
			test_acc = correct/total
			print("Test accuracy (macro): {} \n".format(test_acc))


