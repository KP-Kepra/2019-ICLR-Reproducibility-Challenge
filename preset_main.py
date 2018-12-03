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

import pl, mp
from networks import *

plt.tight_layout()

# pt = torch.load('models/MiniAlexNet-8.pt')
# model = MiniAlexNet()
model = LeNet()
# model.load_state_dict(pt)
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

''' MP ESD FIT '''
def esd():
  for i, module in enumerate(model.modules()):
    # print(i, module)
    if isinstance(module, nn.Linear):
      print(i, module)

      if i == 7:
        W_tensor = module.weight.data.clone().to(device)
        # W_tensor = reshape_tensor(W_tensor)
        W = np.array(W_tensor)  

        M, N = np.min(W.shape), np.max(W.shape)
        Q = N/M

        u, sv, sh = svd(W)

        # Eigenvalues = square of singular values
        evs = sv * sv
        # fit = pl.fit_powerlaw(evs)
        # pl.plot_powerlaw(fit)

        sigma = mp.plot_ESD_MP(evs, Q, 100)
        sr = mp.calc_mp_soft_rank(evals=evs,Q=Q, sigma=sigma)
        print(sr)

''' GENERALIZATION GAP '''
batch_list = [2, 4, 8, 16, 32, 100, 250, 500]
num_spikes = {2:0, 4:52, 8:32, 16:17, 32:9, 50:9, 100:7, 150:5, 250:0, 500:0, 1000:0}
stable_ranks = []
soft_ranks = []

transform = transforms.Compose([
    transforms.Resize(32),
    transforms.ToTensor(), 
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

testset = torchvision.datasets.CIFAR10(
              root='./data', train=False, 
              download=False, transform=transform)

testloader = torch.utils.data.DataLoader(
              testset, batch_size=64, 
              shuffle=False, num_workers=3)

for batch in batch_list:
  model_name = 'models/LeNet-' + str(batch) + '.pt'
  pt = torch.load(model_name)
  model.load_state_dict(pt)
  model.to(device)

  correct = 0
  total = 0

  with torch.no_grad():
    for data in testloader:
      images, labels = data
      images, labels = images.to(device), labels.to(device)
      outputs  = model(images)
      _, predicted = torch.max(outputs.data, 1)
      total += labels.size(0)
      correct += (predicted == labels).sum().item()

  print('Batch size: ', str(batch))
  print('Accuracy of network over 10k test images %d %%' % (
    100 * correct / total ))
  print()