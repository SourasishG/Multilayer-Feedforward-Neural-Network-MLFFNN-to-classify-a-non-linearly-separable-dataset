# Multilayer Feedforward Neural Network (MLFFNN) — Simple Guide

A short, beginner-friendly neural network built from scratch using **NumPy** to solve the non-linear **XOR Classification Problem**.

---

## What is this assignment about?

A single linear line cannot solve the XOR problem:
* Point `(0.5, 0.5)`  $\to$ **Class 1**
* Point `(-0.5, 0.5)` $\to$ **Class 0**
* Point `(-0.5, -0.5)`$\to$ **Class 1**
* Point `(0.5, -0.5)` $\to$ **Class 0**

To solve this, we build a **3-Layer Neural Network**:
`Input (2 features)` $\to$ `Hidden Layer (Non-linear Activation)` $\to$ `Output Layer (Sigmoid Probability)`

---

## How each Task is Solved in `mlffnn.py`

| Task # | Task Description | How It is Done in Code |
| :---: | :--- | :--- |
| **1** | Generate non-linear dataset | Created 200 random points where Class 1 if $x_1 \times x_2 > 0$, else Class 0. |
| **2** | Normalize and split data | Standardized with $(X - \text{mean}) / \text{std}$; split 80% train, 20% test. |
| **3** | Design Neural Network | Initialized weights `W1, b1` (input $\to$ hidden) and `W2, b2` (hidden $\to$ output). |
| **4** | Forward propagation | Computes $Z_1 = X W_1 + b_1$, applies activation $A_1$, then calculates $A_2 = \sigma(A_1 W_2 + b_2)$. |
| **5** | Backpropagation | Uses the chain rule: $dZ_2 = A_2 - y$, updates weights via Gradient Descent. |
| **6** | Train with Gradient Descent | Loops 1000 times (epochs) with learning rate $\alpha = 0.1$. |
| **7** | Experiment hidden neurons | Evaluated `2`, `4`, and `8` hidden neurons. |
| **8** | Compare activations | Tested `ReLU` (best), `Tanh` (good), and `Sigmoid` (slowest). |
| **9** | Plot training loss | Saves a clean loss curve to `training_results.png`. |
| **10**| Final performance evaluation | Evaluates on unseen test set (100% accuracy) and provides a `predict(x1, x2)` helper. |

---

## How to Run

1. Open your terminal in this directory:
   ```bash
   python mlffnn.py
   ```

2. View the output image `training_results.png` created in the folder.

---

## How to Test Your Own Inputs

At the bottom of `mlffnn.py`, simply call `predict(x1, x2)` with any two numbers:

```python
# Pass any two numbers to get the prediction!
c, confidence = predict(0.5, 0.5)
print("Class:", c)
```

* **Class 1**: When both numbers have the same sign (e.g. `(0.5, 0.5)` or `(-0.5, -0.5)`).
* **Class 0**: When the numbers have opposite signs (e.g. `(-0.5, 0.5)` or `(0.5, -0.5)`).
