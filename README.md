# Neural Network From Scratch — Forward Propagation + Backpropagation in NumPy

A minimal, fully manual implementation of a 2-layer neural network — forward pass, loss, backpropagation, and gradient descent — no PyTorch, no TensorFlow, just NumPy and matrix calculus.

This project exists to answer one question: *what is actually happening inside a neural network when it "learns"?*

## Overview

The network pushes an input through two layers to produce a probability distribution, measures how wrong that prediction is, then walks the error backward through the network to update every weight and bias:

```
Input (X)
   ↓
Layer 1: z1 = W1·X + b1   →  ReLU     →  a1
   ↓
Layer 2: z2 = W2·a1 + b2  →  Softmax  →  ŷ
   ↓
Loss = Cross-Entropy(ŷ, Y)
   ↓
Backpropagation: dz2 → dW2, db2 → dA1 → dz1 → dW1, db1
   ↓
Gradient Descent: W -= lr * dW,  b -= lr * db
   ↓
Repeat for 1000 epochs
```

Every gradient is derived and coded by hand — no `.backward()`, no autograd.

## Files

| File | Description |
|---|---|
| `neuralNetwork.py` | Full forward pass + loss + backpropagation + training loop |
| `README.md` | This file |

## How It Works

**1. Forward pass** (same as the earlier forward-propagation-only version — a linear transform, ReLU, another linear transform, Softmax).

**2. Loss — Cross-Entropy**
```python
loss = -np.sum(Y * np.log(Y_hat + 1e-8))
```
Measures how far the predicted probability distribution is from the true one-hot label `Y`. The `+ 1e-8` prevents `log(0)`, which would otherwise blow up to `-inf`.

**3. Backpropagation — the chain rule, applied layer by layer**
```python
dz2 = Y_hat - Y                 # gradient of Softmax + Cross-Entropy combined
dW2 = dz2 @ a1.T
db2 = dz2

dA1 = W2.T @ dz2                # push the error back through W2
dz1 = dA1 * relu_derivative(z1) # push it back through ReLU
dW1 = dz1 @ X.T
db1 = dz1
```
The key trick here is that `dz2 = Y_hat - Y` is the *combined* gradient of Softmax and Cross-Entropy — when these two are paired together (as they almost always are for classification), their gradients simplify beautifully into this one clean subtraction. That's not a coincidence, it's why Softmax and Cross-Entropy are used together so often.

**4. Parameter update — Gradient Descent**
```python
W2 -= learning_rate * dW2
b2 -= learning_rate * db2
W1 -= learning_rate * dW1
b1 -= learning_rate * db1
```
Repeated for 1000 epochs.

## Example Output

```
Epoch    0 | Loss: 0.000123 | pred: 1
Epoch  500 | Loss: 0.000113 | pred: 1
Epoch  999 | Loss: 0.000105 | pred: 1
```

## An Honest Caveat

The loss barely moves here — it drops from `0.000123` to `0.000105` over 1000 epochs. That's not a bug. It's because the hand-picked initial weights already predicted the correct class (`1`) with 99.99% confidence *before training even started*. The network had almost nothing left to learn.

That means this version proves the **mechanics** are correct (the gradients are well-formed, the loss decreases monotonically, nothing diverges) — but it doesn't yet demonstrate *learning* in any visible way, since there's no real "wrong → right" journey to show.

## Requirements

```
numpy
```

## How to Run

```bash
python neuralNetwork.py
```

Note: the current script prints every value on every epoch, which floods the console with 1000 iterations' worth of output. Worth trimming the per-epoch `print()` calls and keeping only the `if epoch % 100 == 0` block before sharing this anywhere.

## What This Project Demonstrates

- How to derive and implement backpropagation manually, layer by layer, using the chain rule
- Why Softmax + Cross-Entropy are paired so often: their combined gradient simplifies to `Y_hat - Y`
- How the gradient of the loss flows backward through each layer (`dz2 → dA1 → dz1`) and gets converted into weight/bias updates
- Why initialization matters: a network that starts "already correct" has nothing meaningful to learn, no matter how many epochs you run

## Next Steps

- **Randomize the initial weights** (e.g. small random values instead of hand-picked ones) so the network actually starts wrong and you can watch the loss drop meaningfully — right now there's no visible "learning curve" to show.
- **Add more training examples** — currently there's exactly one `(X, Y)` pair, so the network is memorizing a single point rather than learning a general function. Train on a small set (e.g. XOR, as the original roadmap suggests) to see real generalization.
- **Plot the loss curve** across epochs (matplotlib) once there's an interesting curve to show — this is the single best way to visually prove "the network learned something."
- **Compare against PyTorch's `autograd`** — feed the same inputs/weights into a PyTorch version, call `.backward()`, and confirm the gradients match your hand-derived ones exactly. That comparison is the real payoff of doing this manually first.
