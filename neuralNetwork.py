import numpy as np
# Declare features 
X = np.array([
    [2],
    [3]
])

# Weight for layer 1
W1 = np.array([
    [1,-1],
    [2,1]
])
#bias for layer 1
b1 = np.array([
    [0],
    [1]
])
#Calculate the output of layer 1
z1=W1@X + b1
print(z1)

#Apply activation function (ReLU) to layer 1 output)
a1 =np.maximum(0,z1)
print(a1)
#weight for layer 2
W2 = np.array([
    [1,0],
    [0,1],
    [1,-1]
])
#bias for layer 2
b2 = np.array([
    [1],
    [2],
    [0]
])
#Calculate the output of layer 2
z2=W2@a1 + b2   
print(z2)

#Apply activation function (Softmax) to layer 2 output'
def softmax(z):
    exp_z = np.exp(z-np.max(z))
    return exp_z / np.sum(exp_z)
Y_hat = softmax(z2)
print(Y_hat)

prediction = np.argmax(Y_hat)
print(prediction)