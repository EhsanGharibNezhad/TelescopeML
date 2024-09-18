#class to demonstrate clustering metrics for the non-outliers
import matplotlib.pyplot as pyplot
import streamlit as st
import numpy as np

from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

class metrics:
    def __init__(self, data, labels):
        non_outliers_mask = labels != -1
        non_outliers_data = data[non_outliers_mask]
        non_outliers_labels = labels[non_outliers_mask]

        if len(set(non_outliers_labels)) < 2:
            print("Not enough clusters to compute scores")
        
        else:
            self.silhouette = silhouette_score(non_outliers_data, non_outliers_labels)
            self.davies_bouldin = davies_bouldin_score(non_outliers_data, non_outliers_labels)
            self.calinski_harabasz = calinski_harabasz_score(non_outliers_data, non_outliers_labels)

    def visualize_metric(metric_name, metric_value, target_value, reverse=False, higher_is_better=True):
        fig, ax = pyplot.subplots()
        
        # Normalize value between 0 and 1
        normalized_value = metric_value / (target_value + 1 * np.exp(-5)) if higher_is_better else 1 - (metric_value / (target_value + 1 * np.exp(-5)))
        normalized_value = min(max(normalized_value, 0), 1)  # Ensure value stays between 0 and 1
        
        # Bar chart to represent the metric value
        ax.barh([metric_name], [metric_value], color='green' if normalized_value >= 0.5 else 'red', height=0.5)
        ax.set_xlim([0, max(metric_value, target_value) * 1.1])  # Slightly bigger than target for spacing
        
        # Displaying target values
        ax.axvline(target_value, color='blue', linestyle='--', label=f'Target: {target_value}')
        ax.set_xlabel('Score')
        ax.legend(loc='lower right')

        # Display chart
        st.pyplot(fig)