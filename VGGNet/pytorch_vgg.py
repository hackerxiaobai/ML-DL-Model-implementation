import torch as t
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

class VGGNet(nn.Module):
	"""docstring for MNIST"""
	def __init__(self):
		super(VGGNet, self).__init__()
		self.conv1_1 = nn.Conv2d(3, 64, 3,stride=1,padding=1)
		self.conv1_2 = nn.Conv2d(64, 64, 3,stride=1,padding=1)
		self.conv2_1 = nn.Conv2d(64, 128, 3,stride=1,padding=1)
		self.conv2_2 = nn.Conv2d(128, 128, 3,stride=1,padding=1)
		self.conv3_1 = nn.Conv2d(128, 256, 3,stride=1,padding=1)
		self.conv3_2 = nn.Conv2d(256, 256, 3,stride=1,padding=1)
		self.conv3_3 = nn.Conv2d(256, 256, 3,stride=1,padding=1)
		self.conv4_1 = nn.Conv2d(256, 512, 3,stride=1,padding=1)
		self.conv4_2 = nn.Conv2d(512, 512, 3,stride=1,padding=1)
		self.conv4_3 = nn.Conv2d(512, 512, 3,stride=1,padding=1)
		self.conv5_1 = nn.Conv2d(512, 512, 3,stride=1,padding=1)
		self.conv5_2 = nn.Conv2d(512, 512, 3,stride=1,padding=1)
		self.conv5_3 = nn.Conv2d(512, 512, 3,stride=1,padding=1)
		self.fc1 = nn.Linear(1*1*512, 4096)
		self.fc2 = nn.Linear(4096, 4096)
		self.fc3 = nn.Linear(4096, 1000)


	def forward(self, x):
		x = F.relu(self.conv1_1(x))
		x = F.relu(self.conv1_2(x))
		x = F.max_pool2d(x,2,2)

		x = F.relu(self.conv2_1(x))
		x = F.relu(self.conv2_2(x))
		x = F.max_pool2d(x,2,2)

		x = F.relu(self.conv3_1(x))
		x = F.relu(self.conv3_2(x))
		x = F.relu(self.conv3_3(x))
		x = F.max_pool2d(x,2,2)

		x = F.relu(self.conv4_1(x))
		x = F.relu(self.conv4_2(x))
		x = F.relu(self.conv4_3(x))
		x = F.max_pool2d(x,2,2)

		x = F.relu(self.conv5_1(x))
		x = F.relu(self.conv5_2(x))
		x = F.relu(self.conv5_3(x))
		x = F.max_pool2d(x,2,2)

		x = F.avg_pool2d(x, 7,1)

		x = x.view(x.size()[0],-1)
		x = F.relu(self.fc1(x))
		x = F.dropout(x,0.5)
		x = F.relu(self.fc2(x))
		x = F.dropout(x,0.5)
		x = self.fc3(x)
		return x


mnist = VGGNet()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(mnist.parameters())


data = t.randn(2, 3, 227, 227)
labels = t.randint(2, size=(2,), dtype=t.int64)

for epoch in range(200):
	optimizer.zero_grad()
	outputs = mnist(data)
	loss = criterion(outputs,labels)
	loss.backward()
	optimizer.step()
	if epoch%10==0:
		print('epoch:%04d,------------loss:%f'%(epoch,loss.item()))
