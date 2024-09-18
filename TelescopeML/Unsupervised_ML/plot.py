import matplotlib.pyplot as pyplot
import numpy as np
import streamlit as st

class plot:
    # Function to visualize 2D scatter plot
    @staticmethod
    def plot_2d_scatter(data, labels, title="2D Scatter Plot"):
        # Ensure labels are not None
        if labels is None or len(labels) == 0:
            raise ValueError("Labels are missing or empty")

        # Get unique labels and ensure they are valid
        unique_labels = np.unique(labels)
        
        if len(unique_labels) == 0:
            raise ValueError("No unique labels found")

        # Find the maximum label, but ensure it's valid
        non_outlier_labels = unique_labels[unique_labels != -1]
        if len(non_outlier_labels) > 0:
            max_label = np.max(non_outlier_labels)
        else:
            max_label = 1  # Fallback in case all points are outliers

        # Get the colormap
        cmap = pyplot.cm.get_cmap('viridis')

        # Plot each cluster
        fig, ax = pyplot.subplots()

        for label in unique_labels:
            if label == -1:
                color = 'k'  # Black color for outliers
                marker = 'x'
                label_name = "Outliers"
            else:
                # Normalize label value to the range [0, 1]
                normalized_value = label / max_label if max_label != 0 else 0
                color = cmap(normalized_value)
                marker = 'o'
                label_name = f"Cluster {label}"

            # Get the data points for the current cluster
            cluster_data = data[labels == label]
            ax.scatter(cluster_data[:, 0], cluster_data[:, 1], c=[color], label=label_name, marker=marker, alpha=0.7, s=5)

        # Add legend, title, and labels
        ax.legend()
        pyplot.title(title)
        pyplot.xlabel("Component 1")
        pyplot.ylabel("Component 2")
        st.pyplot(fig)

    # Function to visualize 3D scatter plot
    @staticmethod
    def plot_3d_scatter(data, labels, title="3D Scatter Plot"):
        fig = pyplot.figure()
        ax = fig.add_subplot(111, projection='3d')
        unique_labels = np.unique(labels)

        # Determine the maximum label for normalization
        max_label = np.max(unique_labels) if unique_labels is not None and len(unique_labels) > 0 else 1
        
        # Get the colormap
        cmap = pyplot.cm.get_cmap('viridis')

        for label in unique_labels:
            if label == -1:
                color = 'k'  # Black color for outliers
                marker = 'x'
                label_name = "Outliers"
            else:
                # Normalize label value to the range [0, 1]
                normalized_value = label / max_label if max_label != 0 else 0
                color = cmap(normalized_value)
                marker = 'o'
                label_name = f"Cluster {label}"

            cluster_data = data[labels == label]

            # Set size of the points (e.g., size 10 for smaller points)
            ax.scatter(cluster_data[:, 0], cluster_data[:, 1], cluster_data[:, 2], 
                       c=[color], label=label_name, marker=marker, alpha=0.7, s=1)  # Adjusted point size

        # Add legend, title, and labels
        ax.legend()
        pyplot.title(title)
        ax.set_xlabel("Component 1")
        ax.set_ylabel("Component 2")
        ax.set_zlabel("Component 3")
        st.pyplot(fig)
