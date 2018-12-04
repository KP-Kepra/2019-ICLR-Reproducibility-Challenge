'''
LeNet5 Structure

0. Input Layer         : 3 channels, 28x28
1. Convolutional Layer : 20 feature maps, 28x28
2. Subsampling Layer   : 20 feature maps, 14x14
3. Convolutional layer : 50 feature maps, 14x14
4. Subsampling layer   : 50 feature maps, 7x7
5. FC Layer            : 500 feature maps
7. Output Layer        : 10 feature maps.
'''

import torch
import torch.nn as nn 

from networks.net_base import Net, Flatten

class LeNet(Net):
  def __init__(self):
    super(LeNet, self).__init__()

    self.conv1 = nn.Conv2d(in_channels = 1, out_channels = 20, kernel_size = 5, padding=1)
    self.conv2 = nn.Conv2d(20, 50, 5, 1) 

    self.flat = Flatten()

    self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    self.fc1 = nn.Linear(in_features = 4 * 4 * 50, out_features = 500)
    self.fc2 = nn.Linear(500, 10)

    self.classifier = nn.Sequential(
      # CONV-1
      self.conv1, nn.ReLU(), self.pool,

      # CONV-2
      self.conv2, nn.ReLU(), self.pool,

      # FC-1
      self.flat, self.fc1, nn.ReLU(),

      # FC-2
      self.fc2
    )
    