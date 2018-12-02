from sklearn.decomposition import TruncatedSVD
import torchvision.models as models
import matplotlib.pyplot as plt
import torch.nn as nn
import torch
import numpy as np
from scipy.linalg import svd
from sklearn.neighbors import KernelDensity

import pl, mp
from networks import *

plt.tight_layout()

pt = torch.load('models/MiniAlexNet-8.pt')
model = MiniAlexNet()
model.load_state_dict(pt)
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