import numpy as np
# Declare features 
X = np.array([
    [2],
    [3]
])
#Add targeted labels
Y =np.array([
    [0],
    [1],
    [0]
])
# Weight for layer 1
W1 = np.array([
    [1,-1],
    [2,1]
], dtype=float)
#bias for layer 1
b1 = np.array([
    [0],
    [1]
], dtype=float)
#weight for layer 2
W2 = np.array([
    [1,0],
    [0,1],
    [1,-1]
], dtype=float)

#bias for layer 2
b2 = np.array([
    [1],
    [2],
    [0]
], dtype=float)

# Activation functions
def relu(z):
    return np.maximum(0,z)
def softmax(z):
    exp_z = np.exp(z-np.max(z))
    return exp_z / np.sum(exp_z)
def relu_derivative(z):
    return (z > 0).astype(float)


#Train
#Update the weights and biases using gradient descent
learning_rate = 0.01
for epoch in range(1000):
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

    prediction = np.argmax(Y_hat)
    print("Prediction",prediction)

    #Calculate the loss
    loss = -np.sum(Y * np.log(Y_hat  + 1e-8))
    print("Loss",loss)
    #backpropagation
    # Calculate the gradient of the loss with respect to the output of layer 2
    dz2 = Y_hat - Y
    # Calculate the gradient of the loss with respect to the weights and biases of layer 2
    dW2 = dz2 @ a1.T
    db2 = dz2
    dA1 = W2.T @ dz2
    # Calculate the gradient of the loss with respect to the output of layer 1
    dz1 = dA1 * relu_derivative(z1)
    dW1 = dz1 @ X.T
    db1 = dz1

    #update 
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2
    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1

    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.4f}")
