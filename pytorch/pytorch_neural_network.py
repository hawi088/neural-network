import torch
import torch.nn as nn

#create tensor for features of XOR
X = torch.tensor([
    [0.0,0.0],
    [0.0,1.0],
    [1.0,0.0],
    [1.0,1.0]
])

#create tensor for labels of XOR
y = torch.tensor([
    [0.0],
    [1.0],
    [1.0],
    [0.0]
])

