print("Script started")
import torch
import torch.nn as nn
torch.manual_seed(42)

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
        self.layer1 = nn.Linear(2,4)
        self.layer2 = nn.Linear(4,2)

    def forward(self,x):
        Z1 = self.layer1(x)
        A1 = torch.relu(Z1)
        Z2 = self.layer2(A1)
        return Z2
model = XORModel()

#calculate the loss
loss_fn = nn.CrossEntropyLoss()

#optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

#training
for epoch in range(1000):
    optimizer.zero_grad() #clear gradients from previous step
    logits = model(X) #run X through the model to get predictions
    loss = loss_fn(logits, y)
    loss.backward()
    optimizer.step() #update the weights based on the gradients
    if epoch % 100 == 0:
        print("Logits:", logits)
        print("Predictions:", torch.argmax(logits, dim=1))
        print("Actual:", y)
        print(epoch, loss.item())

