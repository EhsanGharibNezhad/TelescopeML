import matplotlib
matplotlib.use('Agg')  # Use the 'Agg' backend for non-interactive plotting
import tkinter as tk
from tkinter import ttk
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from TelescopeML.DataMaster import DataProcessor
from Unsupervised_ML import DBSCANProcessor, KMeansClustering, PCAProcessor
import os

# Load and prepare the data
def load_data():
    # Get the reference data path from environment variables
    __reference_data_path__ = os.getenv("reference_data")
    
    # Load the data using the reference data path
    df = pd.read_csv(os.path.join(__reference_data_path__, 
                                  'training_datasets', 
                                  'browndwarf_R100_v4_newWL_v3.csv.bz2'), compression='bz2')

    # Define the output columns
    output_names = ['gravity', 'temperature', 'c_o_ratio', 'metallicity']
    output_values = df[output_names].values

    # Extract input features
    flux_values = df.drop(columns=output_names).values
    wavelength_names = [col for col in df.columns if col not in output_names]
    wavelength_values = [float(item) for item in wavelength_names]

    # Initialize DataProcessor
    data_processor = DataProcessor(
        flux_values=flux_values,
        wavelength_names=wavelength_names,
        wavelength_values=wavelength_values,
        output_values=output_values,
        output_names=output_names
    )

    # Split and standardize the data
    data_processor.split_train_validation_test()
    data_processor.standardize_X_column_wise()
    
    # Include the output values for color mapping
    return data_processor, output_values

# Main application class
class TelescopeMLApp(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title("TelescopeML Unsupervised Learning")
        self.geometry("800x600")

        # Load data
        self.data_processor, self.output_values = load_data()
        self.X = self.data_processor.X_train_standardized_columnwise

        # Choose a feature for the color gradient (e.g., 'temperature')
        self.color_feature_index = 1  # Assuming 'temperature' is the second feature

        # Create widgets
        self.create_widgets()

    def create_widgets(self):
        # Create a dropdown menu for selecting the method
        self.method_label = tk.Label(self, text="Choose Method")
        self.method_label.pack()

        self.method = ttk.Combobox(self, values=['K-Means', 'DBSCAN', 'PCA'])
        self.method.pack()
        self.method.current(0)
        self.method.bind("<<ComboboxSelected>>", self.update_param_widgets)  # Bind the selection event

        # Create parameter inputs
        self.param_frame = tk.Frame(self)
        self.param_frame.pack()

        self.create_param_widgets()

        # Create a button to run the selected method
        self.run_button = tk.Button(self, text="Run", command=self.run_method)
        self.run_button.pack()

        # Create a canvas for plotting
        self.fig, self.ax = plt.subplots(figsize=(5, 4))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.get_tk_widget().pack()

    def create_param_widgets(self):
        # Clear existing parameter widgets
        for widget in self.param_frame.winfo_children():
            widget.destroy()

        # Create parameter widgets based on the selected method
        if self.method.get() == 'K-Means':
            self.k_label = tk.Label(self.param_frame, text="Number of Clusters")
            self.k_label.grid(row=0, column=0)

            self.k_value = tk.IntVar(value=3)
            self.k_entry = tk.Entry(self.param_frame, textvariable=self.k_value)
            self.k_entry.grid(row=0, column=1)

        elif self.method.get() == 'DBSCAN':
            self.eps_label = tk.Label(self.param_frame, text="Epsilon")
            self.eps_label.grid(row=0, column=0)

            self.eps_value = tk.DoubleVar(value=0.5)
            self.eps_entry = tk.Entry(self.param_frame, textvariable=self.eps_value)
            self.eps_entry.grid(row=0, column=1)

            self.min_samples_label = tk.Label(self.param_frame, text="Minimum Samples")
            self.min_samples_label.grid(row=1, column=0)

            self.min_samples_value = tk.IntVar(value=5)
            self.min_samples_entry = tk.Entry(self.param_frame, textvariable=self.min_samples_value)
            self.min_samples_entry.grid(row=1, column=1)

        elif self.method.get() == 'PCA':
            self.n_components_label = tk.Label(self.param_frame, text="Number of Components")
            self.n_components_label.grid(row=0, column=0)

            self.n_components_value = tk.IntVar(value=2)
            self.n_components_entry = tk.Entry(self.param_frame, textvariable=self.n_components_value)
            self.n_components_entry.grid(row=0, column=1)

    def update_param_widgets(self, event):
        """Update parameter widgets based on the selected method."""
        self.create_param_widgets()

    def run_method(self):
        # Clear the plot
        self.ax.clear()

        # Run the selected method
        if self.method.get() == 'K-Means':
            self.run_kmeans()
        elif self.method.get() == 'DBSCAN':
            self.run_dbscan()
        elif self.method.get() == 'PCA':
            self.run_pca()

        # Update the canvas with the new plot
        self.canvas.draw()

    def run_kmeans(self):
        n_clusters = self.k_value.get()
        kmeans_processor = KMeansClustering(n_clusters=n_clusters)
        kmeans_processor.fit(self.X)

        # Plot the clusters
        self.ax.scatter(self.X[:, 0], self.X[:, 1], c=kmeans_processor.labels, cmap='viridis', marker='o', edgecolor='k')
        self.ax.scatter(kmeans_processor.cluster_centers[:, 0], kmeans_processor.cluster_centers[:, 1], s=300, c='red', marker='X')
        self.ax.set_title("K-Means Clustering")
        self.ax.set_xlabel("Feature 1")
        self.ax.set_ylabel("Feature 2")

    def run_dbscan(self):
        eps = self.eps_value.get()
        min_samples = self.min_samples_value.get()
        dbscan_processor = DBSCANProcessor(
            flux_values=self.X,
            eps=eps,
            min_samples=min_samples
        )
        dbscan_processor.fit()

        # Plot the clusters
        unique_labels = set(dbscan_processor.get_labels())
        colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
        for k, col in zip(unique_labels, colors):
            if k == -1:
                # Black used for noise.
                col = [0, 0, 0, 1]

            class_member_mask = (dbscan_processor.labels_ == k)
            xy = self.X[class_member_mask]
            self.ax.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col), markeredgecolor='k', markersize=6)
        
        self.ax.set_title("DBSCAN Clustering")
        self.ax.set_xlabel("Feature 1")
        self.ax.set_ylabel("Feature 2")

    def run_pca(self):
        n_components = self.n_components_value.get()
        pca_processor = PCAProcessor(n_components=n_components)
        X_transformed = pca_processor.fit_transform(self.X)
    
        # Use one of the output values for color mapping (e.g., temperature)
        colors = self.output_values[:self.X.shape[0], self.color_feature_index]
    
        # Normalize the color values to be in the range [0, 1] for matplotlib
        norm = plt.Normalize(vmin=colors.min(), vmax=colors.max())
    
        # Plot the first two principal components with color gradient
        scatter = self.ax.scatter(X_transformed[:, 0], X_transformed[:, 1], c=colors, cmap='viridis', norm=norm)
        self.ax.set_title("PCA")
        self.ax.set_xlabel("Principal Component 1")
        self.ax.set_ylabel("Principal Component 2")
    
        # Add colorbar to the plot
        colorbar = self.fig.colorbar(scatter, ax=self.ax)
        colorbar.set_label('Temperature')  # Label the colorbar appropriately

# # Run the application
# if __name__ == "__main__":
#     app = TelescopeMLApp()
#     app.mainloop()

def main():
    app = TelescopeMLApp()
    app.mainloop()

# If this script is executed directly
if __name__ == "__main__":
    main()

