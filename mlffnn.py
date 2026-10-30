import numpy as np
import matplotlib.pyplot as plt

# 1 & 2. Generate Non-Linear XOR Dataset & Normalize (80/20 Split)
np.random.seed(42)
X = np.random.uniform(-1, 1, (200, 2))
y = ((X[:, 0] * X[:, 1]) > 0).astype(float).reshape(-1, 1)
X = (X - X.mean(axis=0)) / X.std(axis=0)
X_tr, X_te = X[:160], X[160:]
y_tr, y_te = y[:160], y[160:]

# Activation function and derivative (ReLU, Tanh, Sigmoid)
def activate(z, act):
    if act == 'relu':
        return np.maximum(0, z), (z > 0).astype(float)
    elif act == 'tanh':
        a = np.tanh(z)
        return a, 1.0 - a**2
    else:
        a = 1.0 / (1.0 + np.exp(-z))
        return a, a * (1.0 - a)

# 3, 4, 5, 6. MLFFNN Forward, Backprop, and Gradient Descent
def run(h=8, act='relu'):
    W1, b1 = np.random.randn(2, h) * 0.5, np.zeros((1, h))
    W2, b2 = np.random.randn(h, 1) * 0.5, np.zeros((1, 1))
    losses, accs = [], []

    for _ in range(1000):
        # 4. Forward Propagation
        z1 = np.dot(X_tr, W1) + b1
        a1, d1 = activate(z1, act)
        a2 = 1.0 / (1.0 + np.exp(-(np.dot(a1, W2) + b2)))

        losses.append(-np.mean(y_tr * np.log(a2 + 1e-9) + (1 - y_tr) * np.log(1 - a2 + 1e-9)))
        accs.append(np.mean((a2 >= 0.5) == y_tr))

        # 5. Backpropagation & Weight Update
        dz2 = a2 - y_tr
        dz1 = np.dot(dz2, W2.T) * d1
        W2 -= 0.1 * np.dot(a1.T, dz2) / len(X_tr)
        b2 -= 0.1 * np.sum(dz2, axis=0, keepdims=True) / len(X_tr)
        W1 -= 0.1 * np.dot(X_tr.T, dz1) / len(X_tr)
        b1 -= 0.1 * np.sum(dz1, axis=0, keepdims=True) / len(X_tr)

    # 10. Test Set Evaluation
    a1_te, _ = activate(np.dot(X_te, W1) + b1, act)
    test_acc = np.mean((1.0 / (1.0 + np.exp(-(np.dot(a1_te, W2) + b2))) >= 0.5) == y_te)
    return losses, accs, test_acc

# 7. Experiment: Hidden Neurons
print("--- 7. Hidden Neurons ---")
for h in [2, 4, 8]:
    print(f"Neurons {h}: Acc = {run(h=h)[2] * 100:.1f}%")

# 8. Experiment: Activation Functions
print("\n--- 8. Activation Functions ---")
for a in ['relu', 'sigmoid', 'tanh']:
    print(f"Activation {a.upper()}: Acc = {run(act=a)[2] * 100:.1f}%")

# 9 & 10. Train Best Model, Plot, and Final Evaluation
losses, accs, final_acc = run(h=8, act='relu')
print(f"\nFinal Test Accuracy: {final_acc * 100:.2f}%")

plt.figure(figsize=(6, 3))
plt.plot(losses, label='Loss', color='red')
plt.plot(accs, label='Accuracy', color='blue')
plt.title("Training Loss & Accuracy")
plt.legend()
plt.tight_layout()
plt.savefig("training_results.png")
print("Plot saved to 'training_results.png'")
