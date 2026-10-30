import numpy as np
import matplotlib.pyplot as plt

# 1 & 2. Dataset & Normalize (Non-linear XOR, 80/20 Train/Test split)
np.random.seed(42)
X = np.random.uniform(-1, 1, (200, 2))
y = ((X[:, 0] * X[:, 1]) > 0).astype(float).reshape(-1, 1)

# Save mean and std so we can normalize custom inputs later
mean = X.mean(axis=0)
std = X.std(axis=0)
X = (X - mean) / std

X_train, X_test = X[:160], X[160:]
y_train, y_test = y[:160], y[160:]

# 4. Forward Propagation
def forward(X, W1, b1, W2, b2, act):
    z1 = np.dot(X, W1) + b1
    if act == 'relu':
        a1 = np.maximum(0, z1)
    elif act == 'tanh':
        a1 = np.tanh(z1)
    else:
        a1 = 1.0 / (1.0 + np.exp(-z1))
    a2 = 1.0 / (1.0 + np.exp(-(np.dot(a1, W2) + b2)))
    return z1, a1, a2

# 3, 5, 6. Train Network with Gradient Descent
def run_model(h=8, act='relu'):
    W1, b1 = np.random.randn(2, h) * 0.5, np.zeros((1, h))
    W2, b2 = np.random.randn(h, 1) * 0.5, np.zeros((1, 1))
    losses, accs = [], []

    for _ in range(1000):
        # Forward pass & record metrics
        z1, a1, a2 = forward(X_train, W1, b1, W2, b2, act)
        losses.append(-np.mean(y_train * np.log(a2 + 1e-9) + (1 - y_train) * np.log(1 - a2 + 1e-9)))
        accs.append(np.mean((a2 >= 0.5) == y_train))

        # 5. Backpropagation & Weight Update
        dz2 = a2 - y_train
        if act == 'relu':
            d1 = (z1 > 0).astype(float)
        elif act == 'tanh':
            d1 = 1.0 - a1**2
        else:
            d1 = a1 * (1.0 - a1)

        dz1 = np.dot(dz2, W2.T) * d1
        W2 -= 0.1 * np.dot(a1.T, dz2) / len(X_train)
        b2 -= 0.1 * np.sum(dz2, axis=0, keepdims=True) / len(X_train)
        W1 -= 0.1 * np.dot(X_train.T, dz1) / len(X_train)
        b1 -= 0.1 * np.sum(dz1, axis=0, keepdims=True) / len(X_train)

    # 10. Test Accuracy
    _, _, a2_test = forward(X_test, W1, b1, W2, b2, act)
    test_acc = np.mean((a2_test >= 0.5) == y_test)
    return losses, accs, test_acc, (W1, b1, W2, b2)

# 7. Experiment: Hidden Neurons
print("--- 7. Hidden Neurons ---")
for h in [2, 4, 8]:
    _, _, acc, _ = run_model(h=h, act='relu')
    print(f"Neurons {h}: Test Accuracy = {acc * 100:.1f}%")

# 8. Experiment: Activation Functions
print("\n--- 8. Activation Functions ---")
for act in ['relu', 'sigmoid', 'tanh']:
    _, _, acc, _ = run_model(h=8, act=act)
    print(f"Activation {act.upper()}: Test Accuracy = {acc * 100:.1f}%")

# 9 & 10. Plot Loss/Accuracy and Final Evaluation
losses, accs, final_acc, (W1, b1, W2, b2) = run_model(h=8, act='relu')
print(f"\nFinal Test Accuracy (ReLU, 8 Neurons): {final_acc * 100:.2f}%")

plt.figure(figsize=(7, 3.5))
plt.plot(losses, label='Loss', color='red')
plt.plot(accs, label='Accuracy', color='blue')
plt.title("Training Loss & Accuracy")
plt.xlabel("Epoch")
plt.legend()
plt.tight_layout()
plt.savefig("training_results.png")
print("Plot saved to 'training_results.png'")

# ==============================================================
# HOW TO TEST WITH YOUR OWN CUSTOM INPUTS:
# Feel free to change the numbers inside the list below!
# ==============================================================
my_inputs = np.array([
    [ 0.6,  0.7],  # Quadrant 1 (both > 0) -> Expected: Class 1
    [-0.5,  0.8],  # Quadrant 2 (x1<0, x2>0) -> Expected: Class 0
    [-0.3, -0.4],  # Quadrant 3 (both < 0) -> Expected: Class 1
    [ 0.9, -0.2]   # Quadrant 4 (x1>0, x2<0) -> Expected: Class 0
])

# Normalize using the training mean and std
norm_inputs = (my_inputs - mean) / std
_, _, probs = forward(norm_inputs, W1, b1, W2, b2, 'relu')
predictions = (probs >= 0.5).astype(int)

print("\n--- Testing Custom Inputs ---")
for pt, pred, prob in zip(my_inputs, predictions, probs):
    print(f"Input: {pt} -> Predicted Class: {pred[0]} (Probability: {prob[0]:.4f})")
