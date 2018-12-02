import torch
import numpy as np
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import powerlaw

from networks import *

# 32x32 for LeNet
# 28x28 for MiniAlexNet

transform = transforms.Compose([
    transforms.Resize(28),
    transforms.ToTensor(), 
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

batch_size = 4

trainset = torchvision.datasets.CIFAR10(
              root='./data', train=True, 
              download=False, transform=transform)

trainloader = torch.utils.data.DataLoader(
              trainset, batch_size=batch_size, 
              shuffle=True, num_workers=3)

testset = torchvision.datasets.CIFAR10(
              root='./data', train=False, 
              download=False, transform=transform)

testloader = torch.utils.data.DataLoader(
              testset, batch_size=batch_size, 
              shuffle=False, num_workers=3)

# Training data 
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Define the network
# net = LeNet()
net = MiniAlexNet()
net.apply(net.init_weights)
net.to(device)
net.cuda()

# Loss Function and Optimizer
loss_fn = nn.CrossEntropyLoss()
# loss_fn = nn.MSELoss(reduction='sum')

optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)

# Train
for epoch in range(10):
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

    if i % 50 == 49:
      print('[%d, %5d] batch: %d loss: %.3f' %
            (epoch + 1, (i + 1) * batch_size, batch_size, running_loss / 2000))
      running_loss = 0.0

print('Finished Training')

torch.save(net.state_dict(), './models/' + str(net._get_name()) + '-' + str(batch_size) + '.pt')

# Test
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

