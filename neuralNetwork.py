import numpy as np
# Declare features XOR
X = np.array([
    [0, 0,1,1],
    [0, 1,0,1],
], dtype=float)
#Add targeted labels
Y =np.array([
    [1,0,0,1],
    [0,1,1,0]
])

np.random.seed(42) # Set a random seed for reproducibility
# Weight for layer 1
W1 = np.random.randn(4,2) *0.5
#bias for layer 1
b1 = np.random.randn(4,1) *0.5
#weight for layer 2
W2 = np.random.randn(2,4) *0.5
#bias for layer 2
b2 = np.random.randn(2,1) *0.5
    
# Activation functions
def relu(z):
    return np.maximum(0,z)
def softmax(z):
    exp_z = np.exp(
        z - np.max(z, axis=0, keepdims=True)
    )

    return exp_z / np.sum(
        exp_z, axis=0, keepdims=True
    )
def relu_derivative(z):
    return (z > 0).astype(float)


#Train
#Update the weights and biases using gradient descent
learning_rate = 0.01
epochs = 10000
loss_history = []

for epoch in range(epochs):
    #Forward pass
    z1=W1@X + b1
    print("Z1",z1)
    #Apply activation function (ReLU) to layer 1 output)
    a1 =relu(z1)
    print("A1",a1)

    #Calculate the output of layer 2
    z2=W2@a1 + b2   
    print("Z2",z2)

    #Apply activation function (Softmax) to layer 2 output'

    Y_hat = softmax(z2)
    print("Y_hat",Y_hat)

    #Calculate the loss of average for all 4 data points
    loss = -np.mean(
        np.sum(Y * np.log(Y_hat + 1e-8), axis=0)
    )


    loss_history.append(loss)
   
    #backpropagation
    # Calculate the gradient of the loss with respect to the output of layer 2
    dz2 = Y_hat - Y
    # Calculate the gradient of the loss with respect to the weights and biases of layer 2
    dW2 = dz2 @ a1.T / X.shape[1]
    db2 = np.mean(dz2, axis=1, keepdims=True)
    dA1 = W2.T @ dz2
    # Calculate the gradient of the loss with respect to the output of layer 1
    dz1 = dA1 * relu_derivative(z1)
    dW1 = dz1 @ X.T / X.shape[1]
    db1 = np.mean(dz1, axis=1, keepdims=True)

    #update 
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2
    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1

    if epoch % 100 == 0:
        print(f"Epoch {epoch:5d} | Loss: {loss:.6f}")
