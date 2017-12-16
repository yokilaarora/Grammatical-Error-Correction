# Neural Network Model having 3 hidden layers with sigmoid activation
import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, V_s, H1, H2, H3, V_t):
		super(Model, self).__init__()
		self.fc1 = nn.Linear(V_s, H1, bias=True)
		self.fc2 = nn.Linear(H1, H2, bias=True)
		self.fc3 = nn.Linear(H2, H3, bias=True)
		self.fc4 = nn.Linear(H3,V_t, bias=True)
		self.sigmoid = nn.Sigmoid()

    def forward(self,x1):
		out1 = self.sigmoid(self.fc1(x1))
		out2 = self.sigmoid(self.fc2(out1))
		out3 = self.sigmoid(self.fc3(out2))
		out4 = self.sigmoid(self.fc4(out3))
		return out4