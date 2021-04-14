#!/usr/bin/env python

import torch.nn as nn
import torch.optim as optim

pretrained = True
lr = 0.001
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam

