# Multilayer Feedforward Neural Network (MLFFNN) from Scratch

A complete, beginner-friendly implementation of a **Multilayer Feedforward Neural Network (MLFFNN)** built from scratch using **NumPy** and **Matplotlib** to solve non-linearly separable classification problems.

---

## Table of Contents
1. [The Problem: Non-Linear Separability](#the-problem-non-linear-separability)
2. [Why Single-Layer Perceptrons Fail](#why-single-layer-perceptrons-fail)
3. [The Solution: Multilayer Feedforward Neural Network](#the-solution-multilayer-feedforward-neural-network)
4. [Step-by-Step Task Breakdown](#step-by-step-task-breakdown)
   - [Task 1: Generate Dataset](#task-1-generate-non-linearly-separable-dataset)
   - [Task 2: Preprocessing & Normalization](#task-2-preprocess-and-normalize-input-data)
   - [Task 3: Network Architecture Design](#task-3-design-multilayer-feedforward-neural-network)
   - [Task 4: Forward Propagation](#task-4-implement-forward-propagation)
   - [Task 5: Backpropagation & Weight Updating](#task-5-implement-backpropagation-and-weight-updating)
   - [Task 6: Gradient Descent Training](#task-6-train-using-gradient-descent)
   - [Task 7: Hidden Neurons Experiment](#task-7-experiment-with-different-hidden-neurons)
   - [Task 8: Activation Functions Comparison](#task-8-compare-activation-functions)
   - [Task 9: Training Loss and Accuracy Plotting](#task-9-plot-training-loss-and-accuracy)
   - [Task 10: Final Performance Evaluation](#task-10-evaluate-final-classification-performance)
5. [Mathematical Formulations](#mathematical-formulations)
6. [Experimental Results & Insights](#experimental-results--insights)
7. [Project Structure & How to Run](#project-structure--how-to-run)

---

## The Problem: Non-Linear Separability

In machine learning and pattern recognition, a dataset is called **linearly separable** if the classes can be completely divided by a single straight line (in 2D space), a flat plane (in 3D space), or a hyperplane (in higher dimensions). Examples include the logical **AND** and **OR** operations.

However, real-world data is almost always **non-linearly separable**. The simplest and most famous theoretical example of non-linear separability is the **Exclusive-OR (XOR)** problem:

| Input $x_1$ | Input $x_2$ | Target Output $y$ (XOR) |
|:-----------:|:-----------:|:-----------------------:|
|      0      |      0      |            0            |
|      0      |      1      |            1            |
|      1      |      0      |            1            |
|      1      |      1      |            0            |

When plotted in a 2D Cartesian plane:
- Points $(0, 1)$ and $(1, 0)$ belong to **Class 1**.
- Points $(0, 0)$ and $(1, 1)$ belong to **Class 0**.

It is mathematically impossible to draw a single straight line that separates the two classes.

---

## Why Single-Layer Perceptrons Fail

In 1969, Marvin Minsky and Seymour Papert published their book *Perceptrons*, proving that a single-layer perceptron can only learn **linearly separable** functions.

A single-layer perceptron computes:
$$\hat{y} = \sigma(w_1 x_1 + w_2 x_2 + b)$$

The decision boundary occurs where $w_1 x_1 + w_2 x_2 + b = 0$, which is strictly a **straight line**.
- To separate $(0,1)$ and $(1,0)$ from $(0,0)$, the line must pass below $(0,1)$ and $(1,0)$ and above $(0,0)$.
- But $(1,1)$ also falls on the same side as $(0,1)$ and $(1,0)$, meaning $(1,1)$ would be incorrectly classified as 1 instead of 0.

This limitation caused the famous first "AI Winter," until multilayer architectures and the backpropagation algorithm gained prominence.

---

## The Solution: Multilayer Feedforward Neural Network

To solve non-linearly separable problems, we introduce **one or more hidden layers** with **non-linear activation functions**.

1. **Feature Space Warping:** Hidden neurons take linear combinations of the inputs and apply non-linear transformations (e.g., ReLU, Tanh, or Sigmoid). This transforms the input space into a new feature space where the classes become linearly separable.
2. **Output Classification:** The final output neuron can now easily draw a linear hyperplane through the warped hidden representation to separate the classes.
3. **Universal Approximation Theorem:** A feedforward neural network with at least one hidden layer and non-linear activation functions can approximate any continuous function given enough hidden neurons.

---

## Step-by-Step Task Breakdown

### Task 1: Generate Non-Linearly Separable Dataset
We generate 300 two-dimensional synthetic samples distributed across four quadrants:
- $y = 1$ when $x_1 \cdot x_2 > 0$ (Quadrants I & III)
- $y = 0$ when $x_1 \cdot x_2 \le 0$ (Quadrants II & IV)

This forms a continuous, non-linearly separable XOR pattern in $[-1, 1] \times [-1, 1]$.

```python
X = np.random.uniform(-1, 1, (300, 2))
y = ((X[:, 0] * X[:, 1]) > 0).astype(float).reshape(-1, 1)
```

---

### Task 2: Preprocess and Normalize Input Data
Neural networks train significantly faster and more reliably when input features have zero mean and unit variance. This prevents gradients from exploding or oscillating wildly along certain feature axes.

1. **Standardization:**
   $$X_{\text{norm}} = \frac{X - \mu}{\sigma}$$
2. **Train/Test Split:** 80% (240 samples) used for training, 20% (60 samples) reserved for testing generalization.

```python
X = (X - X.mean(axis=0)) / X.std(axis=0)
X_train, X_test = X[:240], X[240:]
y_train, y_test = y[:240], y[240:]
```

---

### Task 3: Design Multilayer Feedforward Neural Network
We construct a 3-layer architecture:
- **Input Layer:** 2 neurons (representing features $x_1$ and $x_2$)
- **Hidden Layer:** $N$ neurons (configurable: 2, 4, 8) with non-linear activation
- **Output Layer:** 1 neuron with Sigmoid activation producing class probabilities $\hat{y} \in [0, 1]$

**Weight Initialization:**
Weights are initialized randomly from a normal distribution scaled by 0.5 to break symmetry, while biases are initialized to zero:
$$W_1 \in \mathbb{R}^{2 \times N}, \quad b_1 \in \mathbb{R}^{1 \times N}$$
$$W_2 \in \mathbb{R}^{N \times 1}, \quad b_2 \in \mathbb{R}^{1 \times 1}$$

---

### Task 4: Implement Forward Propagation
Forward propagation computes the output signal moving from inputs through the hidden layer to the output layer:

1. **Hidden Layer Pre-activation:**
   $$Z_1 = X \cdot W_1 + b_1$$
2. **Hidden Layer Activation:**
   $$A_1 = f(Z_1) \quad (\text{where } f \in \{\text{ReLU}, \text{Tanh}, \text{Sigmoid}\})$$
3. **Output Layer Pre-activation:**
   $$Z_2 = A_1 \cdot W_2 + b_2$$
4. **Output Activation (Sigmoid):**
   $$\hat{y} = A_2 = \sigma(Z_2) = \frac{1}{1 + e^{-Z_2}}$$

---

### Task 5: Implement Backpropagation and Weight Updating
Backpropagation applies the **chain rule of calculus** to compute the gradient of the loss function with respect to every weight and bias.

1. **Binary Cross-Entropy Loss:**
   $$\mathcal{L} = -\frac{1}{m} \sum_{i=1}^m \left[ y^{(i)} \ln(A_2^{(i)}) + (1 - y^{(i)}) \ln(1 - A_2^{(i)}) \right]$$

2. **Output Layer Gradients:**
   $$dZ_2 = A_2 - y$$
   $$\frac{\partial \mathcal{L}}{\partial W_2} = \frac{1}{m} A_1^T \cdot dZ_2, \quad \frac{\partial \mathcal{L}}{\partial b_2} = \frac{1}{m} \sum dZ_2$$

3. **Hidden Layer Gradients:**
   $$dA_1 = dZ_2 \cdot W_2^T$$
   $$dZ_1 = dA_1 \odot f'(Z_1)$$
   $$\frac{\partial \mathcal{L}}{\partial W_1} = \frac{1}{m} X^T \cdot dZ_1, \quad \frac{\partial \mathcal{L}}{\partial b_1} = \frac{1}{m} \sum dZ_1$$

4. **Gradient Descent Updates:**
   $$W = W - \alpha \frac{\partial \mathcal{L}}{\partial W}, \quad b = b - \alpha \frac{\partial \mathcal{L}}{\partial b}$$
   where $\alpha$ is the learning rate (0.1).

---

### Task 6: Train Using Gradient Descent
The network iterates for 1,000 epochs. In each epoch:
1. Complete forward pass over the training data.
2. Calculate and record the epoch loss and training accuracy.
3. Complete backward pass to compute parameter gradients.
4. Update weights and biases via gradient descent.

---

### Task 7: Experiment with Different Hidden Neurons
We evaluate hidden layer capacities: **2, 4, and 8 neurons**:
- **2 Neurons:** The network attempts to form linear partitions. It struggles to enclose both diagonal quadrants, resulting in lower test accuracy (~83.3%).
- **4 Neurons:** Provides enough hyperplanes to isolate the quadrants, converging reliably (~83.3% - 90%).
- **8 Neurons:** Offers ample representational capacity, enabling rapid loss decay and high classification accuracy (~93.3% - 100%).

---

### Task 8: Compare Activation Functions
We compare three fundamental non-linear activations under the same architecture (8 hidden neurons, 1,000 epochs):

1. **ReLU ($f(z) = \max(0, z)$):**
   - *Gradient:* 1 if $z > 0$, else 0.
   - *Result:* Fastest convergence, no vanishing gradients for positive inputs. Reached **100.0% test accuracy**.
2. **Tanh ($f(z) = \tanh(z)$):**
   - *Gradient:* $1 - \tanh^2(z)$.
   - *Result:* Zero-centered, smooth gradient transitions. Reached **91.7% test accuracy**.
3. **Sigmoid ($f(z) = \frac{1}{1 + e^{-z}}$):**
   - *Gradient:* $\sigma(z)(1 - \sigma(z))$.
   - *Result:* Slowest convergence due to gradient saturation away from zero (vanishing gradient effect). Reached **55.0% test accuracy** within 1,000 epochs.

---

### Task 9: Plot Training Loss and Accuracy
The script generates a 3-panel comparative figure saved as `training_results.png`:
- **Subplot 1:** Training Loss and Training Accuracy curves for the baseline model (ReLU, 8 neurons).
- **Subplot 2:** Loss curves comparing 2, 4, and 8 hidden neurons.
- **Subplot 3:** Loss curves comparing ReLU, Sigmoid, and Tanh activation functions.

---

### Task 10: Evaluate Final Classification Performance
The trained network is evaluated on the unseen test set (60 samples):
- **Decision Threshold:** Probability $\ge 0.5 \implies \text{Class 1}$, otherwise $\text{Class 0}$.
- **Performance:**
  - Final Test Accuracy: **100.00%**
  - Loss smoothly decreased below 0.05.

---

## Mathematical Formulations

### Activation Derivatives Reference

| Function | Equation $f(z)$ | Derivative $f'(z)$ |
|:---|:---:|:---:|
| **ReLU** | $\max(0, z)$ | $\begin{cases} 1 & z > 0 \\ 0 & z \le 0 \end{cases}$ |
| **Tanh** | $\frac{e^z - e^{-z}}{e^z + e^{-z}}$ | $1 - f(z)^2$ |
| **Sigmoid** | $\frac{1}{1 + e^{-z}}$ | $f(z) \cdot (1 - f(z))$ |

### Gradient Derivation (Output Layer with BCE + Sigmoid)
For a single sample:
$$\mathcal{L} = -y \ln(a_2) - (1 - y) \ln(1 - a_2)$$
Using the chain rule:
$$\frac{\partial \mathcal{L}}{\partial z_2} = \frac{\partial \mathcal{L}}{\partial a_2} \cdot \frac{\partial a_2}{\partial z_2}$$
$$\frac{\partial \mathcal{L}}{\partial a_2} = -\frac{y}{a_2} + \frac{1 - y}{1 - a_2} = \frac{a_2 - y}{a_2 (1 - a_2)}$$
$$\frac{\partial a_2}{\partial z_2} = a_2 (1 - a_2)$$
Multiplying both terms:
$$\frac{\partial \mathcal{L}}{\partial z_2} = \left( \frac{a_2 - y}{a_2 (1 - a_2)} \right) \cdot \left( a_2 (1 - a_2) \right) = a_2 - y$$

This elegant cancellation simplifies both forward and backpropagation implementations.

---

## Experimental Results & Insights

```text
--- Experiment 1: Number of Hidden Neurons ---
Hidden Neurons: 2 | Test Accuracy: 83.3%
Hidden Neurons: 4 | Test Accuracy: 83.3%
Hidden Neurons: 8 | Test Accuracy: 93.3%

--- Experiment 2: Activation Functions ---
Activation: RELU    | Test Accuracy: 100.0%
Activation: SIGMOID | Test Accuracy: 55.0%
Activation: TANH    | Test Accuracy: 91.7%

--- Final Model Performance ---
Final Test Accuracy: 100.00%
```

### Key Takeaways:
1. **Network Depth and Width:** Increasing hidden neurons improves representational capacity, allowing the network to form smoother, more flexible non-linear decision boundaries.
2. **Choice of Activation:** ReLU consistently outperforms Sigmoid in feedforward networks by preventing gradient saturation, leading to significantly faster and more stable convergence.

---

## Project Structure & How to Run

### Directory Layout
```text
d:\Multilayer Feedforward Neural Network\
│
├── mlffnn.py               # Complete standalone Python implementation
├── README.md               # Detailed project explanation and documentation
└── training_results.png    # Output visualization plots
```

### Requirements
- Python 3.8+
- NumPy
- Matplotlib

Install required packages if not already present:
```bash
pip install numpy matplotlib
```

### Running the Code
Execute the script directly from your terminal:
```bash
python mlffnn.py
```
This runs the full training routine, logs experiment results to the console, and generates the plot file `training_results.png`.

