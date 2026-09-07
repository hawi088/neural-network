# Forward Propagation From Scratch — A Tiny Neural Network in NumPy

A minimal, fully manual implementation of forward propagation through a 2-layer neural network — no PyTorch, no TensorFlow, just NumPy and matrix multiplication.

This project exists to answer one question: *what is actually happening inside a neural network before you ever start training it?*

## Overview

Given a single input vector, the network pushes it through two layers and produces a probability distribution over 3 output classes:

```
Input (X)
   ↓
Layer 1: z1 = W1·X + b1   →  ReLU  →  a1
   ↓
Layer 2: z2 = W2·a1 + b2  →  Softmax  →  ŷ
   ↓
Prediction = argmax(ŷ)
```

Every weight, bias, and activation function is hand-coded — no `model.forward()`, no autograd, no framework doing the matrix math behind the scenes.

## Files

| File | Description |
|---|---|
| `neuralNetwork.py` | Full forward pass: layer 1 (ReLU) → layer 2 (Softmax) → prediction |
| `README.md` | This file |

## How It Works

**1. Input**
```python
X = [[2], [3]]
```
A single example with 2 features.

**2. Layer 1 — Linear transformation + ReLU**
```python
z1 = W1 @ X + b1
a1 = np.maximum(0, z1)     # ReLU: keeps positive values, zeroes out negative ones
```
`W1` is a 2×2 weight matrix, `b1` is a bias vector. ReLU introduces non-linearity — without it, stacking layers would collapse into a single linear transformation, no matter how many layers you add.

**3. Layer 2 — Linear transformation + Softmax**
```python
z2 = W2 @ a1 + b2
Y_hat = softmax(z2)
```
`W2` maps the 2-dimensional hidden layer output to 3 output scores (logits). Softmax then converts those raw scores into a proper probability distribution — all values between 0 and 1, summing to 1.

```python
def softmax(z):
    exp_z = np.exp(z - np.max(z))   # subtracting max for numerical stability
    return exp_z / np.sum(exp_z)
```

**4. Prediction**
```python
prediction = np.argmax(Y_hat)
```
The predicted class is simply whichever output has the highest probability.

## Example Output

With the weights and biases defined in the script:

```
z1 = [[-1], [8]]
a1 = [[0],  [8]]        # ReLU zeroed out the negative value

z2 = [[1], [10], [-8]]
Y_hat ≈ [[0.0001], [0.9999], [0.0000]]

prediction = 1
```

The network is almost entirely confident in class `1` — because after Softmax, one logit (`10`) dominates the other two (`1` and `-8`) by a wide margin.

## Requirements

```
numpy
```

Install with:
```bash
pip install numpy
```

## How to Run

```bash
python neuralNetwork.py
```

## What This Project Demonstrates

- How a neural network layer is really just matrix multiplication plus a bias term
- Why activation functions (ReLU, Softmax) matter — without them, a "deep" network is mathematically no different from a single linear model
- The mechanical difference between a hidden layer (ReLU, used internally) and an output layer (Softmax, used to produce interpretable probabilities)
- How raw, unbounded logits get converted into a real probability distribution
- Why floating-point stability matters — subtracting `np.max(z)` before exponentiating in Softmax prevents overflow on large logits

## What's Deliberately Missing (for now)

This script only computes a **forward pass** — it makes one prediction with fixed, hand-picked weights. It does not:

- Learn from data (no training loop)
- Compute a loss (no comparison against a true label)
- Update weights (no gradient descent, no backpropagation)

That's intentional — this project is step one of two. The next step is deriving backpropagation by hand and using it to actually train these weights instead of hardcoding them.

## Next Steps

- Add a loss function (e.g. cross-entropy, which pairs naturally with Softmax)
- Derive the gradients of that loss with respect to `W1`, `b1`, `W2`, `b2` by hand
- Implement backpropagation and a training loop from scratch
- Compare the hand-derived gradients against PyTorch's `autograd` to confirm they match
