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

# 32x32 for LeNet
# 28x28 for MiniAlexNet

transform = transforms.Compose([
    transforms.Resize(28),
    transforms.ToTensor(), 
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

trainset = torchvision.datasets.CIFAR10(
              root='./data', train=True, 
              download=False, transform=transform)

trainloader = torch.utils.data.DataLoader(
              trainset, batch_size=4, 
              shuffle=True, num_workers=3)

testset = torchvision.datasets.CIFAR10(
              root='./data', train=False, 
              download=False, transform=transform)

testloader = torch.utils.data.DataLoader(
              testset, batch_size=4, 
              shuffle=False, num_workers=3)

# Training data 
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Define the network
# net = LeNet()
net = MiniAlexNet()
net.to(device)
net.cuda()

# Loss Function and Optimizer
loss_fn = nn.CrossEntropyLoss()
# loss_fn = nn.MSELoss(reduction='sum')

optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

# Train
for epoch in range(2):
  running_loss = 0.0
  for i, data in enumerate(trainloader, 0):

    # Get the inputs
    inputs, labels = data
    inputs, labels = inputs.to(device), labels.to(device)
    # print(inputs.shape)
    optimizer.zero_grad()
    output = net(inputs)
    loss = loss_fn(output, labels)
    loss.backward()
    optimizer.step()

    running_loss += loss.item()

    if i % 2000 == 1999:
      print('[%d, %5d] loss: %.3f' %
            (epoch + 1, i + 1, running_loss / 2000))
      running_loss = 0.0

print('Finished Training')

# Test
dataiter = iter(testloader)
images, labels = dataiter.next()
images, labels = images.to(device), labels.to(device)

output = net(images)
_, predicted = torch.max(output, 1)

correct = 0
total = 0
with torch.no_grad():
  for data in testloader:
    images, labels = data
    images, labels = images.to(device), labels.to(device)
    outputs  = net(images)
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum().item()

print('Accuracy of network over 10k test images %d %%' % (
  100 * correct / total ))

