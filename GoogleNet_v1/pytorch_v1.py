import torch as t
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from functools import partial


class GoogleNet_v1(nn.Module):
	"""docstring for MNIST"""
	def __init__(self):
		super(GoogleNet_v1, self).__init__()
		self.conv1x1 = partial(nn.Conv2d,kernel_size=1,stride=1,padding=0)
		self.conv3x3 = partial(nn.Conv2d,kernel_size=3,stride=1,padding=1)
		self.conv5x5 = partial(nn.Conv2d,kernel_size=5,stride=1,padding=2)
		self.conv1_1 = nn.Conv2d(3, 64, 7,stride=2,padding=3)

		self.fc1 = nn.Linear(2048, 1024)
		self.fc2 = nn.Linear(1024, 1000)
		self.fc3 = nn.Linear(1024, 1000)


	def inception_module(self,in_tensor, c1, c3_1, c3, c5_1, c5, pp):
		conv1 = self.conv1x1(in_tensor.size()[1],c1)(in_tensor)
		conv3_1 = self.conv1x1(in_tensor.size()[1],c3_1)(in_tensor)
		conv3 = self.conv3x3(conv3_1.size()[1],c3)(conv3_1)
		conv5_1 = self.conv1x1(in_tensor.size()[1],c5_1)(in_tensor)
		conv5 = self.conv5x5(conv5_1.size()[1],c5)(conv5_1)
		pool_conv = self.conv1x1(in_tensor.size()[1],pp)(in_tensor)
		pool = F.max_pool2d(pool_conv, 3, stride=1, padding=1)
		merged = t.cat([conv1, conv3, conv5, pool],1)
		return merged

	def aux_clf(self,in_tensor):
		avg_pool = F.avg_pool2d(in_tensor,5,3)
		conv = self.conv1x1(avg_pool.size()[1],128)(avg_pool)
		flattened = conv.view(conv.size()[0],-1)
		dense = F.relu(self.fc1(flattened))
		dropout = F.dropout(dense,0.7)
		out = F.softmax(self.fc2(dropout))
		return out

	def forward(self, x):
		x = F.max_pool2d(F.pad(F.relu(self.conv1_1(x)),(1,1,1,1)),3,2)
		x = F.relu(self.conv1x1(64,64)(x))
		x = F.max_pool2d(F.pad(F.relu(self.conv3x3(64,192)(x)),(1,1,1,1)),3,2)
		inception3a = self.inception_module(x, 64, 96, 128, 16, 32, 32)
		inception3b = self.inception_module(inception3a, 128, 128, 192, 32, 96, 64)
		pad3 = F.pad(inception3b, (1,1,1,1))
		pool3 = F.max_pool2d(pad3, 3, 2)
		inception4a = self.inception_module(pool3, 192, 96, 208, 16, 48, 64)
		inception4b = self.inception_module(inception4a, 160, 112, 224, 24, 64, 64)
		inception4c = self.inception_module(inception4b, 128, 128, 256, 24, 64, 64)
		inception4d = self.inception_module(inception4c, 112, 144, 288, 32, 48, 64)
		inception4e = self.inception_module(inception4d, 256, 160, 320, 32, 128, 128)
		pad4 = F.pad(inception4e, (1,1,1,1))
		pool4 = F.max_pool2d(pad4, 3, 2)
		aux_clf1 = self.aux_clf(inception4a)
		aux_clf2 = self.aux_clf(inception4d)
		inception5a = self.inception_module(pool4, 256, 160, 320, 32, 128, 128)
		inception5b = self.inception_module(inception5a, 384, 192, 384, 48, 128, 128)
		pad5 = F.pad(inception5b, (1,1,1,1))
		pool5 = F.max_pool2d(pad5, 3, 2)
		avg_pool_1 = F.max_pool2d(pool5, 4,1)
		avg_pool = avg_pool_1.view(avg_pool_1.size()[0],-1)
		dropout = F.dropout(avg_pool,0.4)
		preds = self.fc3(dropout)

		return preds


mnist = GoogleNet_v1()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(mnist.parameters())


data = t.randn(2, 3, 224, 224)
labels = t.randint(2, size=(2,), dtype=t.int64)

for epoch in range(20):
	optimizer.zero_grad()
	outputs = mnist(data)
	loss = criterion(outputs,labels)
	loss.backward()
	optimizer.step()
	if epoch%10==0:
		print('epoch:%04d,------------loss:%f'%(epoch,loss.item()))
