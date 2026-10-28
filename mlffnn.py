import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

# 1. Generate Non-Linearly Separable Dataset (XOR problem)
X = np.random.uniform(-1, 1, (300, 2))
y = ((X[:, 0] * X[:, 1]) > 0).astype(float).reshape(-1, 1)

# 2. Preprocess and Normalize (Standardize & 80/20 train/test split)
X = (X - X.mean(axis=0)) / X.std(axis=0)
X_train, X_test = X[:240], X[240:]
y_train, y_test = y[:240], y[240:]

# 3, 4, 5, 6: MLFFNN Forward, Backprop, and Gradient Descent
def train(X, y, hidden_dim=8, activation='relu', epochs=1000, lr=0.1):
    W1 = np.random.randn(2, hidden_dim) * 0.5
    b1 = np.zeros((1, hidden_dim))
    W2 = np.random.randn(hidden_dim, 1) * 0.5
    b2 = np.zeros((1, 1))

    losses, accuracies = [], []

    for _ in range(epochs):
        # 4. Forward Propagation
        z1 = np.dot(X, W1) + b1
        if activation == 'relu':
            a1 = np.maximum(0, z1)
            d_act = (z1 > 0).astype(float)
        elif activation == 'tanh':
            a1 = np.tanh(z1)
            d_act = 1.0 - a1**2
        else:
            a1 = 1.0 / (1.0 + np.exp(-np.clip(z1, -500, 500)))
            d_act = a1 * (1.0 - a1)

        z2 = np.dot(a1, W2) + b2
        a2 = 1.0 / (1.0 + np.exp(-np.clip(z2, -500, 500)))

        # Compute Loss & Accuracy
        loss = -np.mean(y * np.log(a2 + 1e-9) + (1.0 - y) * np.log(1.0 - a2 + 1e-9))
        acc = np.mean((a2 >= 0.5) == y)
        losses.append(loss)
        accuracies.append(acc)

        # 5. Backpropagation & Weight Updating (Gradient Descent)
        dz2 = a2 - y
        dW2 = np.dot(a1.T, dz2) / len(X)
        db2 = np.sum(dz2, axis=0, keepdims=True) / len(X)

        dz1 = np.dot(dz2, W2.T) * d_act
        dW1 = np.dot(X.T, dz1) / len(X)
        db1 = np.sum(dz1, axis=0, keepdims=True) / len(X)

        W1 -= lr * dW1
        b1 -= lr * db1
        W2 -= lr * dW2
        b2 -= lr * db2

    return losses, accuracies, (W1, b1, W2, b2)

def predict(X, weights, activation='relu'):
    W1, b1, W2, b2 = weights
    z1 = np.dot(X, W1) + b1
    if activation == 'relu':
        a1 = np.maximum(0, z1)
    elif activation == 'tanh':
        a1 = np.tanh(z1)
    else:
        a1 = 1.0 / (1.0 + np.exp(-np.clip(z1, -500, 500)))
    z2 = np.dot(a1, W2) + b2
    a2 = 1.0 / (1.0 + np.exp(-np.clip(z2, -500, 500)))
    return (a2 >= 0.5).astype(float)

# 7. Experiment: Different Numbers of Hidden Neurons
print("--- Experiment 1: Number of Hidden Neurons ---")
plt.figure(figsize=(14, 4))
plt.subplot(1, 3, 2)
for h in [2, 4, 8]:
    loss, _, w = train(X_train, y_train, hidden_dim=h, activation='relu')
    acc = np.mean(predict(X_test, w, 'relu') == y_test)
    print(f"Hidden Neurons: {h} | Test Accuracy: {acc*100:.1f}%")
    plt.plot(loss, label=f"{h} Neurons")
plt.title("Loss vs Hidden Neurons")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

# 8. Experiment: Compare Activation Functions
print("\n--- Experiment 2: Activation Functions ---")
plt.subplot(1, 3, 3)
for act in ['relu', 'sigmoid', 'tanh']:
    loss, _, w = train(X_train, y_train, hidden_dim=8, activation=act)
    acc = np.mean(predict(X_test, w, act) == y_test)
    print(f"Activation: {act.upper():7s} | Test Accuracy: {acc*100:.1f}%")
    plt.plot(loss, label=act.upper())
plt.title("Loss vs Activation Functions")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

# 9 & 10. Train Best Model, Plot Curve, and Evaluate Test Performance
base_loss, base_acc, best_weights = train(X_train, y_train, hidden_dim=8, activation='relu')

plt.subplot(1, 3, 1)
plt.plot(base_loss, label='Loss', color='red')
plt.plot(base_acc, label='Accuracy', color='blue')
plt.title("Training Loss & Accuracy (ReLU, 8 Neurons)")
plt.xlabel("Epoch")
plt.legend()

plt.tight_layout()
plt.savefig("training_results.png")
print("\nPlots saved to 'training_results.png'")

# 10. Final Evaluation
test_preds = predict(X_test, best_weights, activation='relu')
final_acc = np.mean(test_preds == y_test)
print(f"\n--- Final Model Performance ---")
print(f"Final Test Accuracy: {final_acc * 100:.2f}%")
