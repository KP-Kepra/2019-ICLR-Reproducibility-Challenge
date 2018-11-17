'''
MiniALexNet Structure

0. Input Layer         : 3 channels, 28x28
1. Convolutional Layer : 96 feature maps, 24x24 (kernel 5x5)
2. MaxPool Layer   : 96 feature maps, 8x8 (pool size 3x3)
3. Batch Normalization
4. Convolutional Layer : 256 feature maps, 8x8 (kernel 5x5)
5. MaxPool Layer   : 256 feature maps, 3x3 (kernel 3x3)
6. Batch Normalization
7. Flatten
8. FC Layer            : 384 feature maps
9. FC Layer            : 192 feature maps
10. Output Layer       : 10 classes
'''

import torch 
import torch.nn as nn 
import torch.nn.functional as F

class Flatten(nn.Module):
  def forward(self, input):
    return input.view(input.size(0), -1)

class MiniAlexNet(nn.Module):
  def __init__(self):
    super(MiniAlexNet, self).__init__()

    self.conv1 = nn.Conv2d(in_channels=3, out_channels=96, kernel_size=5, padding=2)
    self.conv2 = nn.Conv2d(96, 256, 5, padding=2)

    self.conv1_bn = nn.BatchNorm2d(96)
    self.conv2_bn = nn.BatchNorm2d(256)

    self.pool = nn.MaxPool2d(kernel_size=(3, 3), stride=(3, 3), padding=1)
    self.flat  = Flatten()
    self.fc1   = nn.Linear(in_features = 4 * 4 * 256, out_features = 384)
    self.fc2   = nn.Linear(384, 192)
    self.fc3   = nn.Linear(192, 10)

  def forward(self, x):
    out = self.conv1_bn(self.pool(F.relu(self.conv1(x))))
    out = self.conv2_bn(self.pool(F.relu(self.conv2(out))))
    out = self.flat(out)
    out = F.relu(self.fc1(out))
    out = F.relu(self.fc2(out))
    out = self.fc3(out)
    return out

  def init_weights(self, m):
    if type(m) == nn.Linear:
      torch.nn.init.xavier_uniform(m.weight)
      m.bias.data.fill_(0.01)


