
import numpy as np
import matplotlib.pyplot as plt
from nnfs.datasets import spiral_data

def create_and_plot_spiral_data(samples, classes):
    X, y = spiral_data(samples=samples, classes=classes)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='brg')
    plt.title('Spiral Data Distribution')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.show()
    return X, y
