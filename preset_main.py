from sklearn.decomposition import TruncatedSVD
import torchvision.models as models
import matplotlib.pyplot as plt
import torch.nn as nn
import torch
import powerlaw
import numpy as np
from scipy.linalg import svd

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

    if i == 8:
      print(i, module)
      W_tensor = module.weight.data.clone().to(device)
      W = np.array(W_tensor)
      W = W / np.linalg.norm(W)
      # WW = np.dot(W.transpose(), W)
      # ev, _ = np.linalg.eig(WW)
      # ev8 = ev
      # print(ev8)
      # print(ev8.shape)
      u, sv, sh = svd(W)
      # svd = TruncatedSVD(n_components=M, n_iter=7, random_state=10)
      # svd.fit(W)
      # svals = svd.singular_values_

      # Eigenvalues = square of singular values
      evals = sv * sv
      print(evals.shape)
      # fit = powerlaw.Fit(evals, xmax=np.max(evals), verbose=False)
      # alpha, D = fit.alpha, fit.D
      plt.hist(evals, bins=100, density=True)

      plt.show()