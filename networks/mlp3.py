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

from networks.net_base import Net, Flatten

class MLP3(Net):
  def __init__(self):
    super(MLP3, self).__init__()

    self.flat = Flatten()

    self.fc1 = nn.Linear(in_features = 28 * 28 * 3, out_features = 512)
    self.fc512 = nn.Linear(512, 512)
    self.fc_out = nn.Linear(512, 10)

    self.classifier = nn.Sequential(
      # Flat
      self.flat,

      # FC 28*28*3 - 512
      self.fc1, nn.ReLU(),

      # FC 512 - 512
      self.fc512, nn.ReLU(),

      # FC 512 - 512
      self.fc512, nn.ReLU(),

      # Out
      self.fc_out
    )