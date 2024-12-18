import numpy as np
import matplotlib.pyplot as plt
import nnfs
from nnfs.datasets import spiral_data
from layers_and_activations import Layer_Dense, Activation_ReLU, Activation_Softmax
from loss_functions import Loss_CategoricalCrossentropy

# Initialize nnfs
nnfs.init()

# Data generation
X, y = spiral_data(samples=100, classes=3)

# Layers and activation setup
dense1 = Layer_Dense(2, 3)
activation1 = Activation_ReLU()

dense2 = Layer_Dense(3, 3)
activation2 = Activation_Softmax()

loss_function = Loss_CategoricalCrossentropy()

# For plotting the loss
losses = []
iterations = 10

# Training loop
for epoch in range(iterations):
    dense1.forward(X)
    activation1.forward(dense1.output)

    dense2.forward(activation1.output)
    activation2.forward(dense2.output)

    loss = loss_function.calculate(activation2.output, y)
    losses.append(loss)
    print(f"Epoch {epoch+1}, Loss: {loss}")

# Plotting the loss graph
plt.figure(figsize=(8, 4))
plt.plot(range(1, iterations + 1), losses, marker='o', label='Loss')
plt.title('Loss Progression')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()
