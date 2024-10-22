import matplotlib.pyplot as plt # Import matplotlib for make plots
import numpy as np # Import numpy for calculate degrees
import math # Import math for make mathematical operations

def plot_neural_network_with_labels(layer_sizes, label=True):
    """
    Draw a neural network diagram with labels for the connections.
    
    Parameters:
    layer_sizes (list): List of integers with the number of neurons in each layer. [Input layer, Hidden layer, Output layer]
    label (bool): Whether to label the connections with their weights.
    """
    
    # Layer sizes
    num_layers = len(layer_sizes)
    
    # Graph setup
    figure, ax = plt.subplots(figsize=(10, 8))
    ax.axis('off')  # Desactivamos los ejes
    
    # Equal aspect ratio to avoid distortion
    ax.set_aspect('equal')

    # definition of the x positions of the layers
    layer_x_positions = np.linspace(0, 1, num_layers) 
    
    # Calculation of the maximum number of neurons
    max_neurons = max(layer_sizes)
    
    # Dictionary to store the positions of the neurons
    neuron_positions = {}
    
    # Draw the neurons and connections
    for i, layer_size in enumerate(layer_sizes):
        # Positions of the neurons in the layer
        layer_y_positions = np.linspace(0, 1, layer_size)
        
        # Handle the case where the layer has only one neuron (center it)
        if layer_size == 1:
            layer_y_positions = [0.5]
        
        neuron_positions[i] = [(layer_x_positions[i], y) for y in layer_y_positions]
        
        # Colors by layer
        colors = ["#378f31", "#b06315", "#941a0f"]

        # Circle draw for each neuron
        for j, (x, y) in enumerate(neuron_positions[i]):
            ax.add_patch(plt.Circle((x, y), 0.03, color=colors[i], fill=True, zorder=2))  # Dibujar neurona
            
            # Assign a label to the neuron
            if i == 0:  # Input layer
                ax.text(x-0.005, y+0.05, f'X{j + 1}', fontsize=10, ha='right')
            elif i == num_layers - 1:  # Output layer
                ax.text(x-0.005, y+0.05, f'O{j + 1}', fontsize=10, ha='left')
            else:  # Hidden layer
                ax.text(x-0.005, y+0.05, f'J{j + 1}', fontsize=10, ha='left')
    
    # Draw the connections between neurons
    for i in range(num_layers - 1):
        for j, (x1, y1) in enumerate(neuron_positions[i]):        # Actual layer
            for k, (x2, y2) in enumerate(neuron_positions[i + 1]): # Next layer
                
                # Verify if there are only one neuron in the next layer to adjust the y2 position
                if len(neuron_positions[i+1]) == 1:
                    y2 = 0.5

                ax.plot([x1, x2], [y1, y2], color=colors[i], lw=0.5, zorder=1)

                if label:
                    if i == 0:  
                        x_label = x1 + (x2 - x1) * 0.25 
                        y_label = y1 + (y2 - y1) * 0.25
                    elif i == num_layers - 2:  
                        x_label = x1 + (x2 - x1) * 0.75 
                        y_label = y1 + (y2 - y1) * 0.75
                    else:
                        continue  # No label for hidden layers
                    
                    # Calculate the angle of the connection
                    angle_rad = math.atan2((y2 - y1), (x2 - x1))  # Angle in radians
                    angle_deg = np.degrees(angle_rad)  # Convert to degrees
                    
                    # Set the color of the label
                    label_color = 'darkgreen' if i == 0 else 'darkred'
                    # Tag the connection with the weight
                    ax.text(x_label, y_label, f'w({k + 1},{j + 1})', fontsize=8, color=label_color, 
                            ha='center', va='center', rotation=angle_deg, rotation_mode='anchor',
                            bbox=dict(facecolor='white', edgecolor='none', alpha=0.9))
    return figure

def plot_error_total_by_epoch(errors_total, last_epoch):
    """
    Draw a graph with the total error by epoch.
    
    Parameters:
    errors_total (list): List with the total error by epoch.
    """
    # Graph setup
    fig, ax = plt.subplots(figsize=(10, 5))
    
    # Grid
    ax.grid(True)

    # Draw the graph
    epochs = [(i * 10) for i in range(1,len(errors_total)-1)]
    epochs.insert(0, 0)
    epochs.append(last_epoch)

    ax.plot(epochs, errors_total, color='darkorange', lw=2, marker='o', markersize=3)
    
    # Set the labels
    ax.set_title('Error total por época')
    ax.set_xlabel('Época')
    ax.set_ylabel('Error Total')
    
    # Return the figure
    return fig
    
import matplotlib.pyplot as plt

def plot_histograms(histogram_data_list):
    """
    Plots histograms for multiple images using a logarithmic Y-axis scale and returns the figure(s).
    
    Args:
        histogram_data_list (list): List of histogram data dictionaries (output from get_histogram_data function).
        
    Returns:
        matplotlib.figure.Figure or list: Either a single figure with all histograms, or a list of figures.
    """
    # Create a subplot for each image's histogram
    fig, axs = plt.subplots(len(histogram_data_list), 1, figsize=(6, len(histogram_data_list) * 4))
    
    # Ensure axs is a list even if there is only one subplot
    if len(histogram_data_list) == 1:
        axs = [axs]
    
    # Loop through the histogram data and plot each histogram
    for idx, hist_data in enumerate(histogram_data_list):
        ax = axs[idx]
        if 'grayscale' in hist_data:
            ax.plot(hist_data['grayscale'], color='black')
            ax.set_title(f'Grayscale Histogram for Image {idx+1}')
        else:
            colors = {'r': 'red', 'g': 'green', 'b': 'blue'}
            for channel, hist in hist_data.items():
                ax.plot(hist, color=colors[channel])
            ax.set_title(f'RGB Histogram for Image {idx+1}')
        
        # Set axis limits and labels
        ax.set_xlim([0, 256])
        ax.set_xlabel('Pixel Value')
        ax.set_ylabel('Frequency')
        
        # Set the y-axis to a logarithmic scale
        ax.set_yscale('log')
    
    # Adjust layout
    plt.tight_layout()
    
    return fig

