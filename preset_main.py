from sklearn.decomposition import TruncatedSVD
import torchvision.models as models
import matplotlib.pyplot as plt
import torch.nn as nn
import torch
import numpy as np
from scipy.linalg import svd

import pl, mp
from networks import *

plt.tight_layout()

pt = torch.load('models/MiniAlexNet.pt')
model = MiniAlexNet()
model.load_state_dict(pt)
# model =  models.alexnet(pretrained=True)

device = torch.device('cuda:0')

for i, module in enumerate(model.modules()):
  if isinstance(module, nn.Linear):
    print(i, module)

    if i == 7:
      W_tensor = module.weight.data.clone().to(device)
      W = np.array(W_tensor)  

      M, N = np.min(W.shape), np.max(W.shape)
      Q = N/M

      u, sv, sh = svd(W)

      # Eigenvalues = square of singular values
      evs = sv * sv
      fit = pl.fit_powerlaw(evs)
      pl.plot_powerlaw(fit)

      evs = evs[evs<10]
      x_min, x_max = 0, np.max(evs)
      sigma = mp.plot_ESD_MP(evs, Q, 100)