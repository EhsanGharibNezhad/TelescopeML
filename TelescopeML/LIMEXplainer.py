import numpy as np
from sklearn.metrics import pairwise_distances
from sklearn.linear_model import LinearRegression


class LIMEXplainer:
    def __init__(self, model, X_test, instance_id, num_segments=50,
                 num_perturbations=200, kernel_width=0.25,
                 perturb_function=None, wavelengths=None):
        """
        Initializes the LIMEXplainer with the model, test data, instance ID,
        and configuration for LIME analysis.
        """
        self.model = model
        self.X_test = X_test
        self.set_instance_by_id(instance_id)
        self.num_segments = num_segments
        self.num_perturbations = num_perturbations
        self.kernel_width = kernel_width
        self.perturb_function = (perturb_function if perturb_function
                                 else self.perturb_segment_mean)
        self.wavelengths = (np.array(wavelengths)
                            if wavelengths is not None else None)
        self.random_perturbations = self.generate_random_perturbations()
        self.perturb_and_store_spectra()
        self.cosine_distances = self.calc_cosine_similarity()
        self.weights = self.compute_weights_from_similarity()
        self.surrogate_models, self.importance_coefficients = (
            self.fit_surrogate_models()
        )
        self.top_segments_list = self.analyze()

    # Initialize Instance
    def set_instance_by_id(self, instance_id):
        """
        Sets the current instance of the spectrum and related data
        based on the given instance ID.

        Parameters:
        - instance_id (int): The ID of the spectrum instance to set.
        """
        self.instance_id = instance_id
        self.spectrum_instance = self.X_test[0][0][instance_id]
        self.min_max_instance = self.X_test[1][0][instance_id]

    def get_spectrum_instance_by_id(self, instance_id):
        """
        Retrieves and sets a spectrum instance by its ID
        from the stored test data.

        Parameters:
        - instance_id (int): The ID of the spectrum instance to retrieve.

        Returns:
        - np.ndarray: The spectrum instance.
        """
        self.set_instance_by_id(instance_id)

        return self.spectrum_instance

    # Perturbation Functions
    @staticmethod
    def perturb_segment_mean(spectrum, start_idx, end_idx):
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

    @staticmethod
    def perturb_segment_total_mean(spectrum, start_idx, end_idx):
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

    @staticmethod
    def perturb_segment_noise(spectrum, start_idx, end_idx):
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

        This function creates a binary matrix where each row
        represents a perturbation, and each column corresponds to
        a segment of the spectrum. A value of '1' indicates the segment
        is active or unchanged, while '0' indicates the segment is
        inactive or altered.

        Returns:
        np.ndarray: A binary matrix representing random perturbations.
        """
        return np.random.binomial(
            1, 0.5,
            size=(self.num_perturbations, self.num_segments))

    # Apply Perturbation
    def perturb_and_store_spectra(self):
        """
        Applies all generated perturbations to the original spectrum
        and stores the perturbed spectra.

        Adjusts for non-uniform wavelength distribution if wavelengths
        are provided.
        """
        self.perturbed_spectra = []
        if self.wavelengths is not None:
            sorted_indices = np.argsort(self.wavelengths)
            sorted_wavelengths = self.wavelengths[sorted_indices]
            segment_boundaries = np.linspace(
                sorted_wavelengths[0], sorted_wavelengths[-1],
                self.num_segments + 1)
            segment_indices = np.searchsorted(
                sorted_wavelengths, segment_boundaries, side='right') - 1
            segment_indices = np.clip(
                segment_indices, 0, len(sorted_wavelengths) - 1)
        else:
            segment_length = len(self.spectrum_instance) // self.num_segments
            segment_indices = [
                i * segment_length for i in range(self.num_segments + 1)]
        for perturbation in self.random_perturbations:
            perturbed_spectrum = self.spectrum_instance.copy()
            for i, active in enumerate(perturbation):
                start_idx = segment_indices[i]
                end_idx = (segment_indices[i + 1]
                           if i + 1 < len(segment_indices)
                           else len(self.spectrum_instance))
                if not active:
                    self.perturb_function(
                        perturbed_spectrum, start_idx, end_idx)
            self.perturbed_spectra.append(perturbed_spectrum)
        return self.perturbed_spectra

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
        Calculates the cosine similarity between each perturbed
        spectrum and the original spectrum.
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

        perturbation_predictions = self.predict_perturbations().reshape(
            self.num_perturbations, -1)
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

        return surrogate_models, segment_importance_coefficients

    # Find Top Segments
    def find_top_segments(self, number_of_top_features=5):
        """
        Identifies the top influential segments for each output
        based on the importance coefficients
        obtained from the class's surrogate models.

        Parameters:
        - number_of_top_features (int): The number of top influential
          segments to identify for each output.

        Returns:
        - list of np.ndarray: A list where each element is an array
          of indices for the top influential segments for a specific output.
        """
        top_influential_segments_per_output = []

        for model in self.surrogate_models:
            coeffs = model.coef_
            top_segments = np.argsort(np.abs(coeffs))[-number_of_top_features:]
            top_influential_segments_per_output.append(top_segments)

        return top_influential_segments_per_output

    # LIME Analyser
    def analyze(self, number_of_top_features=5):
        """
        Performs the complete LIME analysis, returning the top influential
         segments for each output.
        """
        top_segments_list = self.find_top_segments(number_of_top_features)

        return top_segments_list
