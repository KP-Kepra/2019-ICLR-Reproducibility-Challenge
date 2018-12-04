from sklearn.decomposition import TruncatedSVD
import torchvision.transforms as transforms
import torchvision.models as models
import matplotlib.pyplot as plt
import torch.nn as nn
import torchvision
import torch
import numpy as np
from scipy.linalg import svd
from sklearn.neighbors import KernelDensity

from networks import *
from tools import *

# num_spikes = {2:0, 4:52, 8:32, 16:17, 32:9, 50:9, 100:7, 150:5, 250:0, 500:0}
num_spikes = {2:0, 4:0, 8:0, 16:0, 32:0, 50:0, 100:0, 150:0, 250:0, 500:0}

ALEXNET_FC1 = 17
ALEXNET_FC2 = 20
ALEXNET_FC3 = 22

MINIALEXNET_FC1 = 10
MINIALEXNET_FC2 = 12

MLP3_FC1 = 0
MLP3_FC2 = 0
MLP3_FC3 = 0

LENET_FC = 8

# pt = torch.load('models/LeNet-128.pt')

# model = MiniAlexNet()
model = LeNet()
# model = MLP3()

# model =  models.alexnet(pretrained=True)

device = torch.device('cuda:0')


def reshape_tensor(W):
  dims = W.shape
  N = np.max(dims)

  M = 1
  for d in dims:
    M = M*d
  M = int(M / N)
  
  if dims[-1] == N:
    Ws = np.reshape(W, (M, N))
  else:
    Ws = np.reshape(W, (N, M))

  return Ws

# batch_list = [2, 4, 8, 16, 32, 100, 250, 500]

''' MP ESD FIT '''
def esd():
  # for batch in batch_list:
    # pt = torch.load('models/MiniAlexNet-' + str(batch) + '.pt')
    pt = torch.load('models/LeNet.pt')
    # model.load_state_dict(pt)
    for i, module in enumerate(model.modules()):
      print(i, module)
      if isinstance(module, nn.Linear):
        print(i, module)

        if i == LENET_FC:
          W_tensor = module.weight.data.clone().to(device)
          # W_tensor = reshape_tensor(W_tensor)
          W = np.array(W_tensor)  

          M, N = np.min(W.shape), np.max(W.shape)
          Q = N/M

          u, sv, sh = svd(W)

          # Eigenvalues = square of singular values
          evs = sv * sv
          fit = pl.fit_powerlaw(evs)
          pl.plot_powerlaw(fit, '')

          sigma = mp.plot_ESD_MP(evs, Q, 0, '')
          sr = mp.calc_mp_soft_rank(evals=evs,Q=Q, sigma=sigma)
          print("Soft Rank : ", sr)
    plt.show()
esd()

''' GENERALIZATION GAP '''
def generalization_gap():
  stable_ranks = []
  soft_ranks = []

  test_acc_list = []
  train_acc_list = []

  loader = Loader('cifar10', 64, './data', False)

  for batch in batch_list:
    model_name = 'models/MiniAlexNet-' + str(batch) + '.pt'
    pt = torch.load(model_name)
    model.load_state_dict(pt)
    model.to(device)

    correct = 0
    total = 0

    with torch.no_grad():
      for data in loader.trainloader:
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        outputs  = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    train_acc = 100 * correct / total
    train_acc_list.append(train_acc)

    print('Batch size: ', str(batch))
    print('Accuracy of network over 10k train images %d %%' % (
      100 * correct / total ))
    print()

    correct = 0
    total = 0

    with torch.no_grad():
      for data in loader.testloader:
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        outputs  = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    test_acc = 100 * correct / total
    test_acc_list.append(test_acc)

    print('Batch size: ', str(batch))
    print('Accuracy of network over 10k test images %d %%' % (
      100 * correct / total ))
    print()

  plt.title("Generalization Gap")
  plt.plot(train_acc_list, linestyle='--', marker='o', label='Train')
  plt.plot(test_acc_list, linestyle='--', marker='o', label='Test')
  plt.xticks(np.arange(len(batch_list)), batch_list)
  plt.ylabel('Accuracy')
  plt.xlabel('Batch Size')
  plt.legend()
  plt.show()

# generalization_gap()