import torch
import numpy as np
from scipy.linalg import svd

from tools import *
from networks import *

MLP3_FC1 = 5
MLP3_FC2 = 7

device = torch.device('cuda:0')
model = MLP3()

epoch_list = [0, 4, 9, 14, 19, 42, 29, 34, 39, 44, 49]

for epoch in epoch_list:
  model_name = 'models/MLP3/epoch-' + str(epoch) + '.pt'
  pt = torch.load(model_name)
  model.load_state_dict(pt)
  model.to(device)

  for (i, module) in enumerate(model.modules()):

    if i == MLP3_FC1:
      W_tensor = module.weight.data.clone().to(device)
      W = np.array(W_tensor)

      M, N = np.min(W.shape), np.max(W.shape)
      Q = N/M

      u, sv, sh = svd(W)

      evs = sv * sv

      sigma = mp.plot_ESD_MP(evs, Q, 0, epoch)
  print(model_name)
plt.show()