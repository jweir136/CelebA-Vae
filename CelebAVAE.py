import torch
import torch.nn as nn
import torch.nn.functional as fn
import torch.optim as optim
import numpy as np
import pandas as pd
import os

class CelebAVAE(nn.Module):
  def __init__(self):
    self.encoder = nn.Sequential(
      nn.Conv2d(3, 16, 5),
      nn.ReLU(True),
      nn.Conv2d(16, 32, 5),
      nn.ReLU(True),
      nn.Conv2d(32, 64, 5),
      nn.ReLU(True),
      nn.Conv2d(64, 128, 5),
      nn.ReLU(True),
      nn.Flatten()
    )
    
  def encode(self, x):
    return self.encoder(x)

  def forward(self, x):
    return x
