import torch
import numpy as np
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim

from models import *

# def unpickle(file):
#   import pickle
#   with open(file, 'rb') as f:
#     dict = pickle.load(f, encoding='bytes')
#   return dict

# data_dir = 'data/cifar-10-batches-py'

# meta_data_dict = unpickle(data_dir + '/batches.meta')
# print(meta_data_dict)
# cifar_label_names = meta_data_dict[b'label_names']
# cifar_label_names = np.array(cifar_label_names)

# cifar_train_data = None
# cifar_train_filenames = []
# cifar_train_labels = []

# for i in range(1, 6):
#   cifar_train_data_dict = unpickle(data_dir + '/data_batch_{}'.format(i))
#   print(cifar_train_data_dict[b'data'])
#   # if i == 1:
#     # cifar_train_data = cifar_train_data_dict[b'data']
# print(cifar_label_names)

transform = transforms.Compose([
    transforms.ToTensor(), 
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=False, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)
print(trainset)
# Training data 

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# Define the network
net = LeNet()
net.to(device)
net.cuda()

# Loss Function and Optimizer
# loss_fn = nn.CrossEntropyLoss()
loss_fn = nn.MSELoss(reduction='sum')
# optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
learning_rate = 1e-3
# Train
for epoch in range(2):
  running_loss = 0.0
  for i, data in enumerate(trainloader):

    # Get the inputs
    inputs, labels = data
    inputs, labels = inputs.to(device).float(), labels.to(device).float()
    # optimizer.zero_grad()
    net.zero_grad()

    output = net(inputs)
    print(output.size(), labels.size())
    loss = loss_fn(output, labels)

    loss.backward()

    with torch.no_grad():
      for param in net.parameters():
        param.data -= learning_rate * param.grad
    
    running_loss += loss.item()
    if i % 2000 == 0:
      print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
      running_loss = 0.0

print('Finished Training')

# Test
    