# XOR, Round Two — Same Problem, Now with PyTorch

The same XOR problem solved in the earlier NumPy project, rebuilt using PyTorch's `nn.Module`, autograd, and a real optimizer — to see exactly what a framework automates versus what you had to do by hand before.

This isn't about getting a "better" result. It's about recognizing every piece of your manual implementation inside PyTorch's abstractions — and this version fixes a real reproducibility issue found during review (see below).

## Overview

```
X (4 examples, 2 features)
   ↓
nn.Linear(2, 4)     →  ReLU
   ↓
nn.Linear(4, 2)     →  raw logits (NOT softmax'd — see note below)
   ↓
nn.CrossEntropyLoss(logits, y)     →  loss.backward()  →  optimizer.step()
   ↓
Repeat for 1000 epochs
```

## Files

| File | Description |
|---|---|
| `pytorch_neural_network.py` | XOR model, training loop, PyTorch version |
| `README.md` | This file |

## What PyTorch Automated (vs. the NumPy version)

| Step | NumPy version | PyTorch version |
|---|---|---|
| Forward pass | Manual `W1@X + b1`, manual `relu()` | `nn.Linear` + `torch.relu()` — same math, no manual matrix bookkeeping |
| Loss | Manual cross-entropy formula, manual one-hot labels | `nn.CrossEntropyLoss()` — takes raw logits + integer class labels directly |
| Backpropagation | Every gradient derived and coded by hand (`dz2`, `dW2`, `dA1`, `dz1`, `dW1`...) | `loss.backward()` — one line, autograd computes every gradient automatically |
| Weight updates | Manual `W -= learning_rate * dW` for every parameter | `optimizer.step()` — one line, updates every parameter automatically |

Everything on the right side of that table is doing exactly the math already derived by hand in the NumPy version — PyTorch just does it automatically, and scales to arbitrarily large networks without writing a new gradient formula every time a layer is added.

## An Important Gotcha: Don't Apply Softmax Yourself

```python
def forward(self, x):
    Z1 = self.layer1(x)
    A1 = torch.relu(Z1)
    Z2 = self.layer2(A1)
    return Z2          # <-- raw logits, NOT softmax(Z2)
```

`nn.CrossEntropyLoss` expects **raw, un-normalized logits** — it applies `log_softmax` internally itself. Manually applying `softmax()` before passing output to `nn.CrossEntropyLoss` would apply softmax twice, silently producing a wrong (too-flat) loss. This script gets it right by returning raw `Z2`.

This is also why labels are given as **class indices** (`y = torch.tensor([0,1,1,0])`) rather than one-hot vectors, unlike the NumPy version — `nn.CrossEntropyLoss` expects indices, not one-hot encodings.

## Reproducibility Fix (found during review)

An earlier version of this script had no `torch.manual_seed(...)` and used only 2 hidden units — the theoretical minimum needed to solve XOR. Testing across multiple random seeds showed training was a genuine coin flip: some seeds converged to 100% accuracy, others got permanently stuck at exactly `ln(2)` loss (50% accuracy) due to "dead ReLU" units — hidden units that output 0 for every input and therefore receive zero gradient, with no way to recover.

This version fixes both issues:
- `torch.manual_seed(42)` — same starting weights every run, same result every time
- `hidden=4` instead of `2` — gives the network enough spare capacity that a dead unit doesn't block learning entirely

## Results

With the fix in place, training is now clean and reproducible:

| Epoch | Loss | Accuracy |
|---|---|---|
| 0 | 0.6882 | 50% |
| 200 | 0.4306 | 75% |
| 300 | 0.2465 | 100% |
| 600 | 0.0451 | 100% |
| 900 | 0.0188 | 100% |

**Final predictions:**

| Input | True Label | Predicted | Confidence |
|---|---|---|---|
| [0, 0] | 0 | 0 | 98.5% |
| [0, 1] | 1 | 1 | 99.1% |
| [1, 0] | 1 | 1 | 97.7% |
| [1, 1] | 0 | 0 | 98.7% |

Notice the network passes through an intermediate 75% accuracy stage (correctly classifying 3 of 4 points) before finding the full solution around epoch 300 — a visible step in the middle of training, rather than an instant jump from wrong to right.

## Requirements

```
torch
```
```bash
pip install torch
```

## How to Run

```bash
python pytorch_neural_network.py
```

## What This Project Demonstrates

- That `nn.Linear`, `torch.relu()`, `nn.CrossEntropyLoss`, `loss.backward()`, and `optimizer.step()` are the exact same operations as the manually derived forward pass, loss, and backpropagation from the NumPy version — just automated
- Why `nn.CrossEntropyLoss` expects raw logits, not pre-softmax'd output, and why applying softmax yourself before it would be a subtle double-application bug
- Why small networks near their theoretical minimum capacity are sensitive to random initialization, and why that sensitivity disappears with a small amount of extra headroom (2 → 4 hidden units)
- Why `torch.manual_seed()` is not optional for anything you intend to report or share — without it, a result is a coincidence, not a finding
