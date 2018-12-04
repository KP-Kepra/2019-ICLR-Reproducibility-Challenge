import torch.optim as optim
import torch.nn as nn
import torch

from networks import *
from tools import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def create_net(net_name):
  ''' NETWORK ARCHITECTURE INITIALIZATION '''
  if net_name == 'minialexnet': net = MiniAlexNet()
  if net_name == 'lenet':       net = LeNet()
  if net_name == 'mlp3':        net = MLP3()
  net.apply(net.init_weights)
  net.to(device)

  if torch.cuda.is_available():
    net.cuda()

  return net

def create_loss_optim(net, optim_name):
  ''' LOSS AND OPTIMIZER '''

  loss_fn = nn.CrossEntropyLoss()

  optimizer = ''
  if optim_name == 'sgd':      optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
  if optim_name == 'adadelta': optimizer = optim.Adadelta(net.parameters(), lr=0.01)

  return loss_fn, optimizer

def train(net_name, dataset, batch_list, epochs, optim_name, checkpoint):
  for batch in batch_list:
    net = create_net(net_name)
    loss_fn, optimizer = create_loss_optim(net, optim_name)
    loader = Loader(dataset, batch, './data', False)

    # Train
    for epoch in range(10):
      running_loss = 0.0
      for i, data in enumerate(loader.trainloader, 0):

        # Get the inputs
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        # Backprop
        optimizer.zero_grad()
        output = net(inputs)
        loss = loss_fn(output, labels)
        loss.backward()
        optimizer.step()

        # Log
        running_loss += loss.item()

        if i % 50 == 49:
          print('[%d, %5d] batch: %d loss: %.3f' %
                (epoch + 1, (i + 1) * batch, batch, running_loss / 2000))
          running_loss = 0.0

      if checkpoint:
        torch.save(net.state_dict(), './models/' + str(net._get_name()) + '/epoch-' + str(epoch) + '.pt')

    print('Finished Training')
    torch.save(net.state_dict(), './models/' + str(net._get_name()) + '-' + str(batch) + '.pt')

    evaluate(net, loader)

def evaluate(net, loader):
  correct = 0
  total = 0

  with torch.no_grad():
    for data in loader.testloader:
      images, labels = data
      images, labels = images.to(device), labels.to(device)

      outputs  = net(images)
      _, predicted = torch.max(outputs.data, 1)

      total += labels.size(0)
      correct += (predicted == labels).sum().item()

  print('Accuracy of network over 10k test images %d %%' % (
    100 * correct / total ))

