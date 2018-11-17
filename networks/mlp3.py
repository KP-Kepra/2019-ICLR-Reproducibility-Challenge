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
  def __init