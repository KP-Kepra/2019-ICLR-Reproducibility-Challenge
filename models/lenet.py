'''
LeNet5 Structure

0. Input Layer         : 3 channels, 32x32
1. Convolutional Layer : 6 feature maps, 28x28
2. Subsampling Layer   : 6 feature maps, 14x14
3. Convolutional layer : 16 feature maps, 10x10
4. Subsampling layer   : 16 feature maps, 5x5
5. FC Layer            : 120 feature maps
6. FC Layer            : 84 feature maps
7. Output Layer        : 10 feature maps.
'''

import torch
import torch.nn as nn 
import torch.nn.functional as F

class LeNet(nn.Module):
  def __init__(self):
    super(LeNet, self).__init__()

    self.conv1 = nn.Conv2d(in_channels = 3, out_channels = 6, kernel_size = 5)
    self.conv2 = nn.Conv2d(6, 16, 5) 

    self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    self.fc1 = nn.Linear(in_features = 16 * 5 * 5, out_features = 120)
    self.fc2 = nn.Linear(120, 84)
    self.fc3 = nn.Linear(84, 10)

  def forward(self, x):
    out = self.pool(F.relu(self.conv1(x)))
    out = self.pool(F.relu(self.conv2(out)))
    out = out.view(out.size(0), -1)
    out = F.relu(self.fc1(out))
    out = F.relu(self.fc2(out))
    out = self.fc3(out)
    return out