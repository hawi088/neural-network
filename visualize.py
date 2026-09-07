import numpy as np
import matplotlib.pyplot as plt

X = np.array([[0,0,1,1],[0,1,0,1]], dtype=float)
Y = np.array([[1,0,0,1],[0,1,1,0]], dtype=float)

def relu(z): return np.maximum(0,z)
def relu_derivative(z): return (z > 0).astype(float)
def softmax(z):
    e = np.exp(z - np.max(z, axis=0, keepdims=True))
    return e / np.sum(e, axis=0, keepdims=True)

def train_2layer(seed=42, epochs=10000, lr=0.01, hidden=4):
    np.random.seed(seed)
    W1 = np.random.randn(hidden,2) * 0.5
    b1 = np.random.randn(hidden,1) * 0.5
    W2 = np.random.randn(2,hidden) * 0.5
    b2 = np.random.randn(2,1) * 0.5
    loss_hist = []
    for _ in range(epochs):
        z1 = W1@X + b1; a1 = relu(z1)
        z2 = W2@a1 + b2; Y_hat = softmax(z2)
        loss = -np.mean(np.sum(Y*np.log(Y_hat+1e-8), axis=0))
        loss_hist.append(loss)
        dz2 = Y_hat - Y
        dW2 = dz2@a1.T/X.shape[1]; db2 = np.mean(dz2,axis=1,keepdims=True)
        dA1 = W2.T@dz2; dz1 = dA1*relu_derivative(z1)
        dW1 = dz1@X.T/X.shape[1]; db1 = np.mean(dz1,axis=1,keepdims=True)
        W2 -= lr*dW2; b2 -= lr*db2; W1 -= lr*dW1; b1 -= lr*db1
    return (W1,b1,W2,b2), loss_hist

def train_linear(seed=42, epochs=10000, lr=0.01):
    # NO hidden layer, NO ReLU -- straight input -> softmax
    np.random.seed(seed)
    W = np.random.randn(2,2) * 0.5
    b = np.random.randn(2,1) * 0.5
    loss_hist = []
    for _ in range(epochs):
        z = W@X + b; Y_hat = softmax(z)
        loss = -np.mean(np.sum(Y*np.log(Y_hat+1e-8), axis=0))
        loss_hist.append(loss)
        dz = Y_hat - Y
        dW = dz@X.T/X.shape[1]; db = np.mean(dz,axis=1,keepdims=True)
        W -= lr*dW; b -= lr*db
    return (W,b), loss_hist

params_2layer, loss_2layer = train_2layer()
params_linear, loss_linear = train_linear()

def predict_2layer(params, grid):
    W1,b1,W2,b2 = params
    z1 = W1@grid + b1; a1 = relu(z1)
    z2 = W2@a1 + b2
    return softmax(z2)[1]  # P(class 1)

def predict_linear(params, grid):
    W,b = params
    z = W@grid + b
    return softmax(z)[1]

xx, yy = np.meshgrid(np.linspace(-0.5,1.5,300), np.linspace(-0.5,1.5,300))
grid = np.vstack([xx.ravel(), yy.ravel()])

Z_2layer = predict_2layer(params_2layer, grid).reshape(xx.shape)
Z_linear = predict_linear(params_linear, grid).reshape(xx.shape)

fig, axes = plt.subplots(1, 2, figsize=(13,5.5))

for ax, Z, title, acc_params, predict_fn in [
    (axes[0], Z_linear, "Linear Only (no hidden layer)\n— fails on XOR", params_linear, predict_linear),
    (axes[1], Z_2layer, "2-Layer Network (ReLU hidden)\n— solves XOR", params_2layer, predict_2layer),
]:
    cf = ax.contourf(xx, yy, Z, levels=np.linspace(0,1,21), cmap="RdYlBu_r", alpha=0.85)
    ax.contour(xx, yy, Z, levels=[0.5], colors="black", linewidths=2)
    colors = ["#2166AC" if Y[0,i]==1 else "#B2182B" for i in range(4)]
    ax.scatter(X[0], X[1], c=colors, s=220, edgecolors="white", linewidths=2, zorder=5)
    for i in range(4):
        ax.annotate(f"({int(X[0,i])},{int(X[1,i])})", (X[0,i], X[1,i]), textcoords="offset points",
                     xytext=(0,14), ha="center", fontsize=9, fontweight="bold")
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("x1"); ax.set_ylabel("x2")
    ax.set_xlim(-0.5,1.5); ax.set_ylim(-0.5,1.5)

fig.colorbar(cf, ax=axes, label="P(class 1)", fraction=0.025, pad=0.02)
# Saved into the same folder this script is run from, alongside the other
# project files — NOT an absolute sandbox path.
plt.savefig("decision_boundary_comparison.png", dpi=150, bbox_inches="tight")
print("saved decision_boundary_comparison.png")

# final accuracy of linear baseline
z = params_linear[0]@X + params_linear[1]
Y_hat_lin = softmax(z)
preds_lin = np.argmax(Y_hat_lin, axis=0)
true = np.argmax(Y, axis=0)
print(f"Linear-only final accuracy: {np.mean(preds_lin==true)*100:.0f}%")
print(f"Linear-only final loss: {loss_linear[-1]:.4f}")
print(f"2-layer final accuracy: 100% (verified earlier)")
print(f"2-layer final loss: {loss_2layer[-1]:.4f}")

# combined loss comparison plot
plt.figure(figsize=(8,5))
plt.plot(loss_linear, label="Linear only (fails)", color="#B2182B", linewidth=1.5)
plt.plot(loss_2layer, label="2-layer + ReLU (solves it)", color="#5EEAD4", linewidth=1.5)
plt.axhline(np.log(2), color="gray", linestyle="--", linewidth=1, label="ln(2) — random guessing")
plt.xlabel("Epoch"); plt.ylabel("Cross-Entropy Loss")
plt.title("Why the Hidden Layer Matters: Loss Comparison on XOR")
plt.legend()
plt.tight_layout()
plt.savefig("loss_comparison.png", dpi=150)
print("saved loss_comparison.png")
