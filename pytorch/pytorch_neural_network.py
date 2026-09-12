print("Script started")
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
y = torch.tensor([0,1,1,0]) #labels for XOR

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
logits = model(X)
print(logits) #random initialization of weights and biases, so output will be different each time

#calculate the loss
loss_fn = nn.CrossEntropyLoss()
loss = loss_fn(logits, y) #convert y to long type for CrossEntropyLoss
print(loss)
