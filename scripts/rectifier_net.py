############################################
# Rectifier Network                        #
############################################

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class RectifierNetwork(nn.Module):
	def __init__(self,in_dim,out_dim):
		""" Constructor
		Input: in_dim	- Dimension of input vector
			   out_dim	- Dimension of output vector
		"""
		super(RectifierNetwork, self).__init__()
		self.fc1 = nn.Linear(in_dim,out_dim)
		#self.fc2 = nn.Linear(out_dim,1)
		self.sigmoid = torch.nn.Sigmoid()


	def forward(self, inp):
		""" Function for forward pass
		Input:	inp 	- Input to the network of dimension in_dim
		Output: output 	- Output of the network with dimension 1
		"""
		out_intermediate = F.relu(self.fc1(inp))
		output =self.sigmoid(1 - torch.sum(out_intermediate,1))
		return output
