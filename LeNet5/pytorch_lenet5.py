import torch as t
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

class LeNet5(nn.Module):
	"""docstring for MNIST"""
	def __init__(self):
		super(LeNet5, self).__init__()
		self.conv1 = nn.Conv2d(1, 6, 5,stride=1)
		self.conv2 = nn.Conv2d(6, 16, 5)
		self.fc1 = nn.Linear(16*5*5, 120)
		self.fc2 = nn.Linear(120, 84)
		self.fc3 = nn.Linear(84, 10)


	def forward(self, x):
		x = F.max_pool2d(F.relu(self.conv1(x)),2)
		x = F.max_pool2d(F.relu(self.conv2(x)),2)
		x = x.view(x.size()[0],-1)
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = self.fc3(x)
		return x


mnist = LeNet5()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(mnist.parameters())


data = t.randn(1000, 1, 32, 32)
labels = t.randint(2, size=(1000,), dtype=t.int64)

for epoch in range(2000):
	optimizer.zero_grad()
	outputs = mnist(data)
	loss = criterion(outputs,labels)
	loss.backward()
	optimizer.step()
	if epoch%10==0:
		print('epoch:%04d,------------loss:%f'%(epoch,loss.item()))
