# Neural Network From Scratch — Solving XOR with Backpropagation in NumPy

A fully manual 2-layer neural network — forward pass, batched cross-entropy loss, backpropagation, and gradient descent — trained from random initialization to solve XOR. No PyTorch, no TensorFlow, just NumPy and matrix calculus.

XOR is the classic proof that a single-layer (linear) model can't solve every problem — no straight line can separate XOR's four points. A hidden layer with a non-linear activation can. Watching this network learn that boundary from scratch is the whole point of the project.

## Overview

```
Input (X)                              4 examples, 2 features each (XOR truth table)
   ↓
Layer 1: z1 = W1·X + b1   →  ReLU     →  a1        (4 hidden units)
   ↓
Layer 2: z2 = W2·a1 + b2  →  Softmax  →  ŷ          (2 output classes)
   ↓
Loss = mean batch Cross-Entropy(ŷ, Y)
   ↓
Backpropagation: dz2 → dW2, db2 → dA1 → dz1 → dW1, db1   (averaged over the batch)
   ↓
Gradient Descent: W -= lr * dW,  b -= lr * db
   ↓
Repeat for 10,000 epochs
```

Every gradient is derived and coded by hand — no `.backward()`, no autograd.

## Files

| File | Description |
|---|---|
| `neuralNetwork.py` | Full forward pass + batched loss + backpropagation + training loop on XOR |
| `xor_loss_curve.png` | Loss over 10,000 training epochs |
| `README.md` | This file |

## The Data

XOR: the model must learn that the output is `1` when exactly one input is `1`, and `0` otherwise.

| x1 | x2 | XOR output |
|---|---|---|
| 0 | 0 | 0 |
| 0 | 1 | 1 |
| 1 | 0 | 1 |
| 1 | 1 | 0 |

All 4 examples are trained together as a single batch (`X` is 2×4 — 2 features, 4 examples as columns), rather than one example at a time.

## How It Works

**1. Forward pass** — same linear → ReLU → linear → Softmax structure as before, but now every operation is batched across all 4 examples at once using NumPy broadcasting.

**2. Loss — batched Cross-Entropy**
```python
loss = -np.mean(np.sum(Y * np.log(Y_hat + 1e-8), axis=0))
```
`np.sum(..., axis=0)` computes the cross-entropy for each individual example (summed across its 2 class probabilities); `np.mean(...)` then averages that loss across all 4 examples in the batch.

**3. Backpropagation, now with batch-averaged gradients**
```python
dz2 = Y_hat - Y                        # combined Softmax + Cross-Entropy gradient
dW2 = dz2 @ a1.T / X.shape[1]          # averaged over the batch (divide by 4)
db2 = np.mean(dz2, axis=1, keepdims=True)

dA1 = W2.T @ dz2
dz1 = dA1 * relu_derivative(z1)
dW1 = dz1 @ X.T / X.shape[1]           # averaged over the batch
db1 = np.mean(dz1, axis=1, keepdims=True)
```
The `/ X.shape[1]` and `np.mean(..., axis=1)` calls are what make this correct for a batch instead of a single example — every gradient gets averaged across all 4 training examples before it's used to update the weights.

**4. Parameter update — Gradient Descent**, unchanged: `W -= learning_rate * dW`, repeated for 10,000 epochs.

## Results

Starting from small random weights, the network begins essentially guessing (loss ≈ `0.70`, roughly equivalent to 50/50 confidence) and converges to confidently correct predictions on all 4 XOR examples:

| Epoch | Loss | Accuracy |
|---|---|---|
| 0 | 0.7006 | 50% |
| 1,000 | 0.5352 | 100% |
| 4,000 | 0.1101 | 100% |
| 9,000 | 0.0265 | 100% |

**Final predictions:**

| Input | True Label | Predicted | P(class 1) |
|---|---|---|---|
| [0, 0] | 0 | 0 | 0.049 |
| [0, 1] | 1 | 1 | 0.985 |
| [1, 0] | 1 | 1 | 0.985 |
| [1, 1] | 0 | 0 | 0.010 |

The network reaches 100% accuracy by epoch 1,000, then spends the remaining 9,000 epochs sharpening its confidence rather than fixing mistakes — visible as the loss curve flattening out but never fully plateauing.

![XOR training loss curve](xor_loss_curve.png)

## Requirements

```
numpy
matplotlib   # only needed if you plot the loss curve
```

## How to Run

```bash
python neuralNetwork.py
```

**Before running:** the current script still has unconditional `print()` calls for `Z1`, `A1`, `Z2`, and `Y_hat` inside the training loop — that's 4 prints × 10,000 epochs of raw array output. Remove those and keep only the `if epoch % 100 == 0` summary line, or the console becomes unusable.

## What This Project Demonstrates

- Why XOR specifically matters: it's not linearly separable, so it's the standard proof that a hidden layer + non-linear activation is *necessary*, not just nice-to-have
- How to properly batch forward propagation, loss, and backpropagation across multiple training examples at once (rather than one example at a time)
- Why gradients need to be averaged (`/ batch_size` or `np.mean`) across a batch before being applied as a weight update
- What real learning looks like end-to-end: starting near-random (loss ≈ `ln(2)` ≈ 0.69, the theoretical loss of guessing with no information) and converging toward a confident, correct solution

