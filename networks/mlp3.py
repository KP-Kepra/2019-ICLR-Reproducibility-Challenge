'''
MLP3 Structure

0. Input Layer    : 3 channels, 28x28
1. Flatten
2. Hidden Layer 1 : 512
3. Hidden Layer 2 : 512
4. Hidden layer 3 : 512
5. Output Layer   : 10
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

class Flatten(nn.Module):
  def forward(self, input):
    return input.view(input.size(0), -1)

class MLP3(nn.Module):
  def __init__(self):
    super(MLP3, self).__init__()

    self.flat = Flatten()

    self.classifier = nn.Sequential(
      self.flat,
      nn.Linear(28 * 28 * 3, 512),
      nn.ReLU(),
      nn.Linear(512, 512),
      nn.ReLU(),
      nn.Linear(512, 512),
      nn.ReLU(),
      nn.Linear(512, 10)
    )

  def forward(self, x):
    out = self.classifier(x)
    return out

  def init_weights(self, m):
    torch.nn.init.xavier_normal_(m.weight)
    m.bias.data.fill_(0.1)