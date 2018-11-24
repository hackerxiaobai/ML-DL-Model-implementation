import torch as t
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

class AlexNet(nn.Module):
	"""docstring for MNIST"""
	def __init__(self):
		super(AlexNet, self).__init__()
		self.conv1 = nn.Conv2d(3, 96, 11,stride=4)
		self.conv2 = nn.Conv2d(96, 256, 5,stride=1,padding=2)
		self.conv3 = nn.Conv2d(256, 384, 3,stride=1,padding=1)
		self.conv4 = nn.Conv2d(384, 256, 3,stride=1,padding=1)
		self.fc1 = nn.Linear(6*6*256, 4096)
		self.fc2 = nn.Linear(4096, 4096)
		self.fc3 = nn.Linear(4096, 1000)


	def forward(self, x):
		x = F.max_pool2d(F.relu(self.conv1(x)),3,2)
		x = F.max_pool2d(F.relu(self.conv2(x)),3,2)
		x = F.relu(self.conv3(x))
		x = F.max_pool2d(F.relu(self.conv4(x)),3,2)
		x = x.view(x.size()[0],-1)
		x = F.relu(self.fc1(x))
		x = F.dropout(x,0.5)
		x = F.relu(self.fc2(x))
		x = F.dropout(x,0.5)
		x = self.fc3(x)
		return x


mnist = AlexNet()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(mnist.parameters())


data = t.randn(10, 3, 227, 227)
labels = t.randint(2, size=(10,), dtype=t.int64)

for epoch in range(200):
	optimizer.zero_grad()
	outputs = mnist(data)
	loss = criterion(outputs,labels)
	loss.backward()
	optimizer.step()
	if epoch%10==0:
		print('epoch:%04d,------------loss:%f'%(epoch,loss.item()))
