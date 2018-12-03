import torch
import torch.nn as nn

class Flatten(nn.Module):
  def forward(self, input):
    return input.view(input.size(0), -1)

class Net(nn.Module):
  def __init__(self):
    super(Net, self).__init__()
    self.classifier = nn.Sequential()

  def init_weights(self, m):
    if type(m) == nn.Linear or type(m) == nn.Conv2d:
      torch.nn.init.xavier_normal_(m.weight)
      m.bias.data.fill_(0.1)

  def forward(self, x):
    out = self.classifier(x)
    return out