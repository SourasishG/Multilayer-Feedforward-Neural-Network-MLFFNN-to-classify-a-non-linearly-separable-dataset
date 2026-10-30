# Multilayer Feedforward Neural Network (MLFFNN)

A compact, beginner-friendly neural network built from scratch using **NumPy** to classify the non-linearly separable **XOR Problem**.

---

## 10 Tasks Implemented in `mlffnn.py`

1. **Dataset**: 200 synthetic XOR points where Class 1 if $x_1 \times x_2 > 0$, else Class 0.
2. **Preprocessing**: Normalized with zero mean and unit variance; split into 80% train, 20% test.
3. **Architecture**: 2 input neurons, $h$ hidden neurons, and 1 sigmoid output neuron.
4. **Forward Propagation**: $Z_1 = X W_1 + b_1 \to A_1 = f(Z_1) \to A_2 = \sigma(A_1 W_2 + b_2)$.
5. **Backpropagation**: Output error $dZ_2 = A_2 - y$ and hidden error $dZ_1 = (dZ_2 \cdot W_2^T) \odot f'(Z_1)$.
6. **Gradient Descent**: Updates $W$ and $b$ across 1,000 epochs with learning rate 0.1.
7. **Hidden Neurons Experiment**: Compares 2, 4, and 8 hidden neurons.
8. **Activation Functions**: Compares ReLU, Sigmoid, and Tanh.
9. **Plot**: Saves the training loss and accuracy curves to `training_results.png`.
10. **Evaluation**: Evaluates final generalization performance on the unseen test set (100% accuracy).

---

## How to Run

```bash
python mlffnn.py
```
Output plot will be generated as `training_results.png`.
