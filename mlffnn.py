import numpy as np
import matplotlib.pyplot as plt

# ==============================================================
# Task 1 & 2: Generate Non-Linear XOR Dataset and Normalize
# ==============================================================
np.random.seed(42)
X = np.random.uniform(-1, 1, (200, 2))
y = ((X[:, 0] * X[:, 1]) > 0).astype(float).reshape(-1, 1)

# Standardization: (X - mean) / std
mean = X.mean(axis=0)
std = X.std(axis=0)
X = (X - mean) / std

# 80% Train, 20% Test split
X_train, X_test = X[:160], X[160:]
y_train, y_test = y[:160], y[160:]

# ==============================================================
# Task 3 & 4: Forward Propagation
# ==============================================================
def forward(X_input, W1, b1, W2, b2, act):
    z1 = np.dot(X_input, W1) + b1

    # Hidden layer activation
    if act == 'relu':
        a1 = np.maximum(0, z1)
    elif act == 'tanh':
        a1 = np.tanh(z1)
    else:
        a1 = 1.0 / (1.0 + np.exp(-z1))  # sigmoid

    # Output layer (always Sigmoid for probability between 0 and 1)
    z2 = np.dot(a1, W2) + b2
    a2 = 1.0 / (1.0 + np.exp(-z2))
    return z1, a1, a2

# ==============================================================
# Task 5 & 6: Backpropagation and Training with Gradient Descent
# ==============================================================
def run_model(neurons=8, act='relu'):
    # Initialize weights and biases
    W1 = np.random.randn(2, neurons) * 0.5
    b1 = np.zeros((1, neurons))
    W2 = np.random.randn(neurons, 1) * 0.5
    b2 = np.zeros((1, 1))

    losses = []
    for _ in range(1000):
        # 1. Forward Pass
        z1, a1, a2 = forward(X_train, W1, b1, W2, b2, act)

        # 2. Compute Loss (Binary Cross-Entropy)
        loss = -np.mean(y_train * np.log(a2 + 1e-9) + (1 - y_train) * np.log(1 - a2 + 1e-9))
        losses.append(loss)

        # 3. Backpropagation (Chain Rule)
        dz2 = a2 - y_train
        if act == 'relu':
            d1 = (z1 > 0).astype(float)
        elif act == 'tanh':
            d1 = 1.0 - a1**2
        else:
            d1 = a1 * (1.0 - a1)

        dz1 = np.dot(dz2, W2.T) * d1

        # 4. Update Weights (Gradient Descent)
        W2 -= 0.1 * np.dot(a1.T, dz2) / len(X_train)
        b2 -= 0.1 * np.sum(dz2, axis=0, keepdims=True) / len(X_train)
        W1 -= 0.1 * np.dot(X_train.T, dz1) / len(X_train)
        b1 -= 0.1 * np.sum(dz1, axis=0, keepdims=True) / len(X_train)

    # Test Accuracy
    _, _, a2_test = forward(X_test, W1, b1, W2, b2, act)
    test_acc = np.mean((a2_test >= 0.5) == y_test)
    return losses, test_acc, (W1, b1, W2, b2)

# ==============================================================
# Task 7: Experiment with Different Hidden Neurons
# ==============================================================
print("--- Task 7: Effect of Hidden Neurons ---")
for n in [2, 4, 8]:
    _, acc, _ = run_model(neurons=n, act='relu')
    print(f"Hidden Neurons: {n} -> Test Accuracy: {acc * 100:.1f}%")

# ==============================================================
# Task 8: Compare Activation Functions (ReLU vs Sigmoid vs Tanh)
# ==============================================================
print("\n--- Task 8: Compare Activation Functions ---")
for act in ['relu', 'sigmoid', 'tanh']:
    _, acc, _ = run_model(neurons=8, act=act)
    print(f"Activation: {act.upper():7s} -> Test Accuracy: {acc * 100:.1f}%")

# ==============================================================
# Task 9: Train Final Model & Plot Training Loss Curve
# ==============================================================
losses, final_acc, weights = run_model(neurons=8, act='relu')
W1, b1, W2, b2 = weights

plt.figure(figsize=(6, 3))
plt.plot(losses, color='red', linewidth=2)
plt.title("Task 9: Training Loss over 1000 Epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.tight_layout()
plt.savefig("training_results.png")
print("\n--- Task 9: Plot saved to 'training_results.png' ---")

# ==============================================================
# Task 10: Final Evaluation & Easy Custom Input Testing
# ==============================================================
print(f"--- Task 10: Final Model Test Accuracy: {final_acc * 100:.2f}% ---")

def predict(x1, x2):
    """Pass any two numbers to get the network's prediction!"""
    pt = np.array([[x1, x2]])
    pt_norm = (pt - mean) / std
    _, _, a2 = forward(pt_norm, W1, b1, W2, b2, 'relu')
    confidence = a2[0, 0]
    predicted_class = int(confidence >= 0.5)
    return predicted_class, confidence

# Try it with custom numbers:
print("\nTesting Custom Inputs with predict(x1, x2):")
for x1, x2 in [(0.5, 0.5), (-0.5, 0.5), (-0.5, -0.5), (0.5, -0.5)]:
    c, conf = predict(x1, x2)
    print(f"predict({x1:4.1f}, {x2:4.1f}) -> Predicted Class: {c}  (Confidence: {conf*100:5.1f}%)")
