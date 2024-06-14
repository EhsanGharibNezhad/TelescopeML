import os
import sys
import logging
import numpy as np
import pandas as pd
from IPython.display import display
from sklearn.metrics import pairwise_distances
from sklearn.linear_model import LinearRegression
from .StatVisAnalyzer import (plot_perturbed_spectrum,
                              plot_most_influential_segments,
                              plot_all_influential_segments,
                              plot_ave_influential_segments,
                              plot_average_influence_per_output,
                              plot_overall_average_influence)


class LIMEXplainer:
    def __init__(self, model, data, features=None):
        """
        Initializes the LIMEXplainer with specified configurations for LIME.

        Parameters:
        - model: Trained machine learning model.
        - data: Dataset containing input data and min-max scaled data.
        - features (np.ndarray, optional): Array of wavelengths corresponding
          to spectrum data points.
        """
        self.model = model
        self.X_test = data
        self.wavelengths = features
        self.instance_index = None
        # Default kernel width for calculating weights
        self.kernel_width = 0.25
        # Default perturbation function
        self.perturb_function = self.perturb_segment_mean
        # Default number of segments
        self.num_segments = 10
        # Default number of perturbations
        self.num_perturbations = 200
        self.spectrum_instance = None
        self.min_max_instance = None
        self.perturbed_spectra = None
        self.cosine_distances = None
        self.weights = None
        self.surrogate_models = None
        self.segment_importance_coefficients = None
        self.last_perturbation_index = -1

    def set_instance_index(self, instance_index):
        """Set a new instance index and update related attributes."""
        if not (0 <= instance_index < len(self.X_test[0][0])):
            raise ValueError("instance_index is out of the range of the data "
                             "provided.")
        self.instance_index = instance_index
        self.spectrum_instance = self.X_test[0][0][self.instance_index]
        self.min_max_instance = self.X_test[1][0][self.instance_index]

    def set_num_segments(self, num_segments):
        """Set a new number of segments and regenerate perturbations."""
        if num_segments is not None:
            if num_segments < 2:
                raise ValueError("Number of segments must be at least 2.")
            elif num_segments > len(self.spectrum_instance):
                raise ValueError(f"Number of segments must not exceed "
                                 f"{len(self.spectrum_instance)}, "
                                 f"which is the number of features "
                                 f"in the dataset.")
        self.num_segments = num_segments
        self.generate_random_perturbations()
        self.perturbed_spectra = self.perturb_and_store_spectra()
        self.weights = self.compute_weights_from_similarity()
        self.surrogate_models, self.segment_importance_coefficients =\
            self.fit_surrogate_models()

    def set_num_perturbations(self, num_perturbations):
        """Set a new number of perturbations and regenerate perturbations."""
        if num_perturbations is not None:
            if num_perturbations < self.num_segments:
                raise ValueError("Number of perturbations must be at least "
                                 "equal to the number of segments.")
            elif num_perturbations > 2 ** self.num_segments:
                raise ValueError(f"Number of perturbations must not exceed "
                                 f"{2 ** self.num_segments}, which is the "
                                 f"maximum possible unique perturbations for "
                                 f"{self.num_segments} segments.")
        self.num_perturbations = num_perturbations
        self.generate_random_perturbations()
        self.perturbed_spectra = self.perturb_and_store_spectra()
        self.weights = self.compute_weights_from_similarity()
        self.surrogate_models, self.segment_importance_coefficients =\
            self.fit_surrogate_models()

    def set_perturb_function(self, perturb_function):
        """Set a new perturbation function."""
        self.perturb_function = perturb_function
        self.perturbed_spectra = self.perturb_and_store_spectra()
        self.weights = self.compute_weights_from_similarity()
        self.surrogate_models, self.segment_importance_coefficients =\
            self.fit_surrogate_models()

    def set_kernel_width(self, kernel_width):
        """Set a new kernel width and recompute weights."""
        self.kernel_width = kernel_width
        self.weights = self.compute_weights_from_similarity()
        self.surrogate_models, self.segment_importance_coefficients =\
            self.fit_surrogate_models()

    def configure_explainer(self, instance_index=None, num_segments=None,
                            num_perturbations=None, perturb_function=None,
                            kernel_width=None,
                            show_segments_with_perturbed_spectrum=False):
        """
        Configure multiple settings of the explainer at once.

        Parameters:
        - instance_index (int, optional): Index of the instance to explain.
        - num_segments (int, optional): Number of segments to divide
          the spectrum into.
        - num_perturbations (int, optional): Number of perturbations
          to generate.
        - perturb_function (callable, optional): Function to perturb segments.
        - kernel_width (float, optional): Width of the kernel
          for calculating weights.
        """
        # Suppress verbose output
        logging.getLogger().setLevel(logging.WARNING)
        original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

        try:
            if instance_index is not None:
                self.set_instance_index(instance_index)
            if num_segments is not None:
                self.set_num_segments(num_segments)
            if num_perturbations is not None:
                self.set_num_perturbations(num_perturbations)
            if perturb_function is not None:
                self.set_perturb_function(perturb_function)
            if kernel_width is not None:
                self.set_kernel_width(kernel_width)
            if show_segments_with_perturbed_spectrum:
                plot_perturbed_spectrum(self.spectrum_instance,
                                        self.get_last_perturbed_spectrum(),
                                        self.wavelengths,
                                        self.num_segments,
                                        self.get_last_perturbation())
        finally:
            # Restore original logging level and stdout
            logging.getLogger().setLevel(logging.INFO)
            sys.stdout.close()
            sys.stdout = original_stdout

    # Perturbation Functions
    def perturb_segment_mean(self, spectrum, start_idx, end_idx):
        """
        Returns a new spectrum array where a segment of the original spectrum
        is replaced with its own mean value.

        Parameters:
        - spectrum (np.ndarray): The original spectrum.
        - start_idx (int): The starting index of the segment to be perturbed.
        - end_idx (int): The ending index of the segment to be perturbed.

        Returns:
        - np.ndarray: New spectrum array with the specified segment perturbed.
        """
        if end_idx > start_idx:
            segment_mean = np.mean(spectrum[start_idx:end_idx])
            spectrum[start_idx:end_idx] = segment_mean
        return spectrum

    def perturb_segment_total_mean(self, spectrum, start_idx, end_idx):
        """
        Returns a new spectrum array where a segment of the original spectrum
        is replaced with the total mean value of the entire spectrum.

        Parameters:
        - spectrum (np.ndarray): The original spectrum.
        - start_idx (int): The starting index of the segment to be perturbed.
        - end_idx (int): The ending index of the segment to be perturbed.

        Returns:
        - np.ndarray: New spectrum array with the specified segment
        replaced by the total mean.
        """
        spectrum[start_idx:end_idx] = np.mean(spectrum)
        return spectrum

    def perturb_segment_noise(self, spectrum, start_idx, end_idx):
        """
        Returns a new spectrum array where a segment of the original spectrum
        is replaced with random noise generated within
        the range of the spectrum's minimum and maximum values.

        Parameters:
        - spectrum (np.ndarray): The original spectrum.
        - start_idx (int): The starting index of the segment to be perturbed.
        - end_idx (int): The ending index of the segment to be perturbed.

        Returns:
        - np.ndarray: New spectrum array with the specified segment
        replaced by random noise.
        """
        noise = np.random.uniform(
            low=np.min(spectrum),
            high=np.max(spectrum),
            size=(end_idx - start_idx)
        )
        spectrum[start_idx:end_idx] = noise
        return spectrum

    # Generate Perturbation
    def generate_random_perturbations(self):
        """
        Generates random perturbations for segments of a spectrum.

        Parameters:
        - num_perturbations (int): The number of perturbations to generate.

        Returns:
        - np.ndarray: A binary matrix representing random perturbations.
        """
        self.random_perturbations = \
            np.random.binomial(1, 0.5, size=(self.num_perturbations,
                               self.num_segments))
        return self.random_perturbations

    def perturb_and_store_spectra(self):
        """
        Generates perturbed spectra by applying the specified perturbation
        function to each segment of the original spectrum. The perturbed
        spectra are stored in the `self.perturbed_spectra` attribute.

        For each perturbation pattern, the function copies the original
        spectrum and perturbs each segment based on the active/inactive state
        in the perturbation pattern. The perturbation function is applied to
        inactive segments.

        Returns:
        - list: A list of perturbed spectra (each as a numpy array).

        Raises:
        - ValueError: If the number of segments is not properly defined.
        """
        self.perturbed_spectra = []
        num_points = len(self.spectrum_instance)
        segment_length = num_points // self.num_segments
        segment_indices = ([i * segment_length for
                            i in range(self.num_segments)]
                           )
        segment_indices.append(num_points)

        for perturbation in self.random_perturbations:
            perturbed_spectrum = self.spectrum_instance.copy()
            for i, active in enumerate(perturbation):
                if not active:
                    start_idx = segment_indices[i]
                    end_idx = segment_indices[i + 1]
                    # Apply the perturbation function
                    self.perturb_function(perturbed_spectrum,
                                          start_idx, end_idx)
            self.perturbed_spectra.append(perturbed_spectrum)
        return self.perturbed_spectra

    def get_last_perturbation(self):
        # Returns the last perturbation used or generated
        return self.random_perturbations[self.last_perturbation_index]

    def get_last_perturbed_spectrum(self):
        # Returns the last perturbed spectrum
        return self.perturbed_spectra[self.last_perturbation_index]

    # Predict Perturbation
    def predict_perturbations(self):
        """
        Predicts the output for each perturbed spectrum
        and collects the predictions.

        Returns:
        - np.ndarray: An array of model predictions for
          each perturbed spectrum.
        """
        perturbation_predictions = []
        for perturbed_spectrum in self.perturbed_spectra:
            len_perturbed = len(perturbed_spectrum)
            perturbed_spectrum_reshaped = perturbed_spectrum.reshape(
                1, len_perturbed, 1)
            input_features = perturbed_spectrum_reshaped.reshape(1, -1)
            min_max_features = self.min_max_instance.reshape(1, -1)
            model_prediction = self.model.predict(
                [input_features, min_max_features])
            perturbation_predictions.append(model_prediction)

        return np.array(perturbation_predictions)

    # Calculate Similarity
    def calc_cosine_similarity(self):
        """
        Calculates the cosine similarity between the original spectrum and
        each perturbed spectrum.

        Returns:
        - np.ndarray: A 1D array of cosine similarity scores between the
        original spectrum and each perturbed spectrum. Each score ranges from
        0 to 1, where 1 indicates identical directionality.
         """
        original_spectrum_reshaped = self.spectrum_instance.reshape(1, -1)
        perturbed_spectra_reshaped = np.vstack(
            [spectrum.reshape(1, -1) for spectrum in self.perturbed_spectra])
        cosine_similarities = 1 - pairwise_distances(
                perturbed_spectra_reshaped, original_spectrum_reshaped,
                metric='cosine')

        return cosine_similarities.flatten()

    # Calculate Weights
    def compute_weights_from_similarity(self):
        """
        Calculates weights for each perturbation using
        an exponential kernel function applied to the cosine similarities.

        Returns:
        np.ndarray: An array of weights for each perturbation,
        derived from the cosine similarities.
        """
        cosine_similarities = self.calc_cosine_similarity()
        weights = np.exp(-(1 - cosine_similarities) ** 2
                         / self.kernel_width ** 2)
        return weights

    # Surrogate Models
    def fit_surrogate_models(self):
        """
        Fits separate linear regression models for each output
        of the model to explain the influence of input features
        using the internally stored perturbations and weights.

        Returns:
        - List of fitted LinearRegression models: One for each output
          of the multi-output model.
        - List of np.ndarray: Each containing the feature importance
          coefficients for one of the outputs.
        """
        surrogate_models = []
        segment_importance_coefficients = []

        perturbation_predictions = self.predict_perturbations().\
            reshape(self.num_perturbations, -1)
        num_outputs = perturbation_predictions.shape[1]

        for output_index in range(num_outputs):
            explainable_model = LinearRegression()
            target_predictions = perturbation_predictions[:, output_index]
            weights = self.compute_weights_from_similarity()
            explainable_model.fit(self.random_perturbations,
                                  target_predictions,
                                  sample_weight=weights
                                  )
            surrogate_models.append(explainable_model)
            segment_importance_coefficients.append(explainable_model.coef_)

        self.surrogate_models = surrogate_models
        self.segment_importance_coefficients = segment_importance_coefficients
        return self.surrogate_models, self.segment_importance_coefficients

    # Find Top Segments
    def analyze_segment_influence(self, top_segments_to_display=None,
                                  show_top_influential_segments=False,
                                  print_top_influential_segments=False,
                                  show_all_segments_influence=False,
                                  show_average_segments_influence=False,
                                  return_data=True):
        """
        Analyzes and visualizes the influence of different segments based on
        the regression coefficients from surrogate models.

        Parameters:
        - top_segments_to_display (int): The number of top influential segments
        to identify for each output.If None, defaults to the number of segments
        - show_top_influential_segments (bool): Whether to display a plot of
        the top influential segments.
        - print_top_influential_segments (bool): Whether to print the dataframe
        of the top influential segments.
        - show_all_segments_influence (bool): Whether to display a plot showing
        the influence of all segments.
        - show_average_segments_influence (bool): Whether to display a plot
        showing the average influence of all segments.
        - return_data (bool): Whether to return the dataframe of top
        influential segments.

        Returns:
        - pd.DataFrame: A dataframe containing the top influential segments,
        their indices, regression coefficients, and wavelength ranges.
        Returned only if return_data is True.
        """
        # Use default number of segments if not specified
        top_segments_to_display = (top_segments_to_display
                                   if top_segments_to_display is not None
                                   else self.num_segments)

        if top_segments_to_display > self.num_segments:
            raise ValueError(f"top_segments_to_display "
                             f"({top_segments_to_display}) "
                             f"cannot be greater than num_segments "
                             f"({self.num_segments}).")
        self.number_of_top_features = top_segments_to_display

        top_influential_segments_per_output = []
        data_rows = []  # List to hold data for DataFrame
        segment_length = len(self.wavelengths) / self.num_segments

        for index, model in enumerate(self.surrogate_models):
            coeffs = model.coef_
            indices_of_top_segments = (np.argsort(-np.abs(coeffs))
                                       [:self.number_of_top_features])
            top_influential_segments_per_output.append(indices_of_top_segments)
            top_segments_coeffs = coeffs[indices_of_top_segments]

            # Calculate wavelength ranges for each segment
            wavelength_ranges = [
                (self.wavelengths[int(seg * segment_length)],
                 self.wavelengths[min(int((seg + 1) * segment_length) - 1,
                                  len(self.wavelengths) - 1)])
                for seg in indices_of_top_segments
            ]

            # Prepare data for DataFrame
            data_row = {'Output': index + 1,
                        'Top Segments Indices': indices_of_top_segments,
                        'Regression Coefficients': top_segments_coeffs,
                        'Wavelength Ranges': wavelength_ranges
                        }
            data_rows.append(data_row)

        # Create DataFrame
        top_influential_segments_per_output_dataframe = pd.DataFrame(data_rows)

        if show_top_influential_segments:
            plot_most_influential_segments(self.spectrum_instance,
                                           self.wavelengths,
                                           self.number_of_top_features,
                                           top_influential_segments_per_output,
                                           self.num_segments
                                           )

        if print_top_influential_segments:
            print('------ Top Influential Segments DataFrame Example ------')
            # Apply formatting when displaying
            display(top_influential_segments_per_output_dataframe.style.format(
                {'Regression Coefficients': lambda x: ['{:.2f}'.format(i)
                                                       for i in x]}))

        if show_all_segments_influence:
            plot_all_influential_segments(self.spectrum_instance,
                                          self.wavelengths,
                                          self.num_segments,
                                          self.segment_importance_coefficients
                                          )

        if show_average_segments_influence:
            plot_ave_influential_segments(self.spectrum_instance,
                                          self.wavelengths,
                                          self.num_segments,
                                          self.segment_importance_coefficients
                                          )

        if return_data:
            return top_influential_segments_per_output_dataframe
        else:
            return None

    # Analyze Multiple Instances
    def analyze_multiple_instances(self, number_of_instances,
                                   print_average_segments_influence=False,
                                   show_all_segments_influence=False,
                                   show_average_segments_influence=False):
        """
        Analyzes and returns the average influence of different segments
        across multiple instances.

        Parameters:
        - number_of_instances (int): Number of instances to analyze.
        - print_average_segments_influence (bool): Whether to print
        the dataframe of average segment influence.
        - show_all_segments_influence (bool): Whether to display plots showing
        the influence of all segments for each output.
        - show_average_segments_influence (bool): Whether to display
        plots showing the average influence of all segments across outputs.

        Returns:
        - pd.DataFrame: A dataframe containing the average regression
        coefficients for each segment across all instances, for each output,
        and the overall average across all outputs.
        """
        # Suppress verbose output
        logging.getLogger().setLevel(logging.WARNING)
        original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

        try:
            instance_indices = list(range(number_of_instances))

            # Initialize array to hold coefficients for all instances
            all_coefficients = []

            for instance_index in instance_indices:
                self.configure_explainer(instance_index=instance_index)
                self.fit_surrogate_models()
                instance_coefficients = []

                for model in self.surrogate_models:
                    instance_coefficients.append(model.coef_)

                all_coefficients.append(instance_coefficients)

            # Convert to numpy array for easier averaging
            all_coefficients = np.array(all_coefficients)

            # Calculate average coefficients across all instances
            average_coefficients = np.mean(all_coefficients, axis=0)

            # Prepare dataframe rows
            data_rows = []
            num_outputs = average_coefficients.shape[0]

            for output_index in range(num_outputs):
                for segment_index in range(self.num_segments):
                    data_row = {
                        'Output': output_index + 1,
                        'Segment': segment_index + 1,
                        'Average Coefficient':
                        average_coefficients[output_index, segment_index]
                    }
                    data_rows.append(data_row)

            # Calculate overall average coefficients across all outputs
            overall_average = np.mean(average_coefficients, axis=0)

            for segment_index in range(self.num_segments):
                data_row = {
                    'Output': 'Overall Average',
                    'Segment': segment_index + 1,
                    'Average Coefficient': overall_average[segment_index]
                }
                data_rows.append(data_row)

            average_segments_df = pd.DataFrame(data_rows)

            if show_all_segments_influence:
                plot_average_influence_per_output(average_coefficients,
                                                  self.num_segments,
                                                  self.wavelengths)

            if show_average_segments_influence:
                plot_overall_average_influence(overall_average,
                                               self.num_segments,
                                               self.wavelengths)

            if print_average_segments_influence:
                print('------ Average Segments Influence DataFrame ------')
                display(average_segments_df)

            return average_segments_df  # Ensure the dataframe is returned

        finally:
            # Restore original logging level and stdout
            logging.getLogger().setLevel(logging.INFO)
            sys.stdout.close()
            sys.stdout = original_stdout
