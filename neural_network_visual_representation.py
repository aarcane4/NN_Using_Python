import matplotlib.pyplot as plt
import numpy as np

def draw_neural_net(ax, left, right, bottom, top, layer_sizes):
    layer_colors = ['skyblue', 'lightgreen', 'salmon', 'violet']  # Colors for each layer
    v_spacing = (top - bottom) / float(max(layer_sizes))
    h_spacing = (right - left) / float(len(layer_sizes) - 1)
    node_radius = v_spacing / 4.  # Radius of the circles representing the neurons

    # Draw nodes
    for n, layer_size in enumerate(layer_sizes):
        layer_top = v_spacing * (layer_size - 1) / 2. + (top + bottom) / 2.
        for m in range(layer_size):
            circle = plt.Circle((n * h_spacing + left, layer_top - m * v_spacing), node_radius,
                                color=layer_colors[n % len(layer_colors)], ec='k', zorder=4)
            ax.add_artist(circle)
            if n == len(layer_sizes) - 1:
                ax.annotate(f'Output\nNeuron {m+1}', (n * h_spacing + left, layer_top - m * v_spacing),
                            textcoords="offset points", xytext=(10,-10), ha='center', color='midnightblue')
            elif n == 0:
                ax.annotate(f'Input {m+1}', (n * h_spacing + left, layer_top - m * v_spacing),
                            textcoords="offset points", xytext=(-10,-10), ha='center', color='midnightblue')

    # Draw edges
    for n, (layer_size_a, layer_size_b) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
        layer_top_a = v_spacing * (layer_size_a - 1) / 2. + (top + bottom) / 2.
        layer_top_b = v_spacing * (layer_size_b - 1) / 2. + (top + bottom) / 2.
        for m in range(layer_size_a):
            for o in range(layer_size_b):
                line = plt.Line2D([n * h_spacing + left, (n + 1) * h_spacing + left],
                                  [layer_top_a - m * v_spacing, layer_top_b - o * v_spacing],
                                  c='gray', alpha=0.5)  # Use gray lines with partial transparency for connections
                ax.add_artist(line)

fig, ax = plt.subplots(figsize=(12, 8))
ax.axis('off')
draw_neural_net(ax, 0.1, 0.9, 0.1, 0.9, [3, 5, 5, 2])
plt.show()


