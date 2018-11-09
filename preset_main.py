from sklearn.decomposition import TruncatedSVD
import torchvision.models as models
import matplotlib.pyplot as plt
import torch.nn as nn
import torch
import powerlaw
import numpy as np

plt.tight_layout()

pretrained_model =  models.alexnet(pretrained=True)

device = torch.device('cuda:0')

for i, module in enumerate(pretrained_model.modules()):
  if isinstance(module, nn.Linear):
    if i == 22:
      W_tensor = module.weight.data.clone().to(device)
      W = np.array(W_tensor)
      M, N = np.min(W.shape), np.max(W.shape)

      svd = TruncatedSVD(n_components=M, n_iter=7, random_state=10)
      svd.fit(W)
      svals = svd.singular_values_

      # Eigenvalues = square of singular values
      evals = (1/N)*svals*svals
      fit = powerlaw.Fit(evals, xmax=np.max(evals), verbose=False)
      alpha, D = fit.alpha, fit.D
      plt.hist(evals, bins=200, density=True)

      plt.show()
print(pretrained_model)