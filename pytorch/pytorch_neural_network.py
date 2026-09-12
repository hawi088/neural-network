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

class XORModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(2,2)
        self.layer2 = nn.Linear(2,2)

    def forward(self,x):
        Z1 = self.layer1(x)
        A1 = torch.relu(Z1)
        Z2 = self.layer2(A1)
        return Z2
model = XORModel()