from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
##########################################################################################
##########################################################################################
for is_CNN in [True, False]:
	
	if is_CNN:
		pretrained_model = "MECHmnist_cnn.pt"
	else:
		pretrained_model = "MECHmnist_fnn.pt"

	use_cuda=False
	##########################################################################################
	##########################################################################################
	# load model 
	##########################################################################################
	class MechMNISTDataset(Dataset):
		""" mechanical MNIST data set"""
	
		def __init__(self,train=True,transform=None, target_transform=None):
			self.train = train
			if self.train:
				self.data = np.load('NPY_FILES/MNIST_bitmap_train.npy')
				self.data = self.data.reshape((self.data.shape[0],28,28))
				self.targets = np.load('NPY_FILES/final_psi_train.npy') 
			else:
				self.data = np.load('NPY_FILES/MNIST_bitmap_test.npy')
				self.data = self.data.reshape((self.data.shape[0],28,28))
				self.targets = np.load('NPY_FILES/final_psi_test.npy')
			self.transform = transform
			self.target_transform = target_transform
		
		def __len__(self):
			return self.data.shape[0]
	
		def __getitem__(self,idx):
			if torch.is_tensor(idx):
				idx = idx.tolist()
		
			img = self.data[idx,:,:]
			lab = self.targets[idx]
		
			#img = Image.fromarray(img.numpy(), mode='L')

			if self.transform is not None:
				img = self.transform(img)

			if self.target_transform is not None:
				lab = self.target_transform(lab)
		
			sample = (img,lab)
		
			return sample 
		
	##########################################################################################
	##########################################################################################
	if is_CNN:
		class Net(nn.Module): 
			def __init__(self):
				super(Net, self).__init__()
				self.conv1 = nn.Conv2d(1, 20, 5, 1)
				self.conv2 = nn.Conv2d(20, 50, 5, 1)
				self.fc1 = nn.Linear(4*4*50, 500)
				self.fc2 = nn.Linear(500, 1) 

			def forward(self, x):
				x = F.relu(self.conv1(x))
				x = F.max_pool2d(x, 2, 2)
				x = F.relu(self.conv2(x))
				x = F.max_pool2d(x, 2, 2)
				x = x.view(-1, 4*4*50)
				x = F.relu(self.fc1(x))
				x = self.fc2(x)
				return x 
	else:
		class Net(nn.Module):
			def __init__(self):
				super(Net, self).__init__()
				self.fc1 = nn.Linear(784, 1568*4)
				self.fc2 = nn.Linear(1568*4,1568)
				self.fc3 = nn.Linear(1568,784)
				self.fc4 = nn.Linear(784, 1)

			def forward(self, x):
				x = x.view(x.shape[0], -1)
				x = F.relu(self.fc1(x))
				x = F.relu(self.fc2(x))
				x = F.relu(self.fc3(x))
				x = F.relu(self.fc4(x))
				return x

	##########################################################################################
	##########################################################################################

	# MNIST Test dataset and dataloader declaration
	train_loader = DataLoader( MechMNISTDataset(train=True, transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))])), batch_size=1, shuffle=False)
	test_loader = DataLoader( MechMNISTDataset(train=False, transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))])), batch_size=1, shuffle=False)

	# Define what device we are using
	print("CUDA Available: ",torch.cuda.is_available())
	device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")

	# Initialize the network
	model = Net().to(device)

	# Load the pretrained model
	model.load_state_dict(torch.load(pretrained_model, map_location='cpu'))

	# Set the model in evaluation mode. In this case this is for the Dropout layers
	model.eval()

	##########################################################################################
	##########################################################################################
	def return_test_all( model, device, test_loader):
		all_test = [] 
		all_target = []
	
		for data, target in test_loader:
			# Send the data and label to the device
			data, target = data.to(device), target.to(device)
			output = model(data)
			output = model(data).detach().numpy()[0][0]
			target = target.detach().numpy()[0]
			all_test.append(output)
			all_target.append(target)
	
		return all_test, all_target

	##########################################################################################
	##########################################################################################
	all_test, all_target = return_test_all( model, device, test_loader)

	if is_CNN:
		np.savetxt('cnn_psitest_correct.txt',np.asarray(all_target))
		np.savetxt('cnn_psitest_predict.txt',np.asarray(all_test))
	else:
		np.savetxt('fnn_psitest_correct.txt',np.asarray(all_target))
		np.savetxt('fnn_psitest_predict.txt',np.asarray(all_test))
	
	all_test, all_target = return_test_all( model, device, train_loader)
	
	if is_CNN:
		np.savetxt('cnn_psitrain_correct.txt',np.asarray(all_target))
		np.savetxt('cnn_psitrain_predict.txt',np.asarray(all_test))
	else:
		np.savetxt('fnn_psitrain_correct.txt',np.asarray(all_target))
		np.savetxt('fnn_psitrain_predict.txt',np.asarray(all_test))


