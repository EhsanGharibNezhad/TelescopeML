import os
import pickle as pk
from tensorflow.keras.models import model_from_json

class LoadSave:
    """
    Load and Save trained operators, models, and datasets
    """

    def __init__(self, ml_model_str, is_feature_improved, is_augmented, is_tuned):
        self.ml_model_str = ml_model_str
        self.is_feature_improved = is_feature_improved
        self.is_augmented = is_augmented
        self.is_tuned = is_tuned
        self.base_path = '../../outputs/regression/trained_models/'

    def create_generic_path(self, indicator):
        """
        Create the generic path for saving or loading the trained model

        Inputs:
        -------
        - indicator (str): Indicator for the model type

        Returns:
        --------
        - generic_path (str): The generic path for saving or loading the model
        """
        file_name = f'{indicator}__{self.ml_model_str}__feature_improved_{self.is_feature_improved}__aug_{self.is_augmented}.tuned__{self.is_tuned}'
        generic_path = os.path.join(self.base_path, file_name)
        return generic_path

    def load_or_dump_trained_object(self, trained_object, indicator, load_or_dump='dump'):
        """
        Load or save the trained object

        Inputs:
        -------
        - trained_object : The object to be saved or loaded
        - indicator (str): Indicator for the type of trained object
        - load_or_dump (str): 'dump' or 'load'
        """
        generic_path = self.create_generic_path(indicator)

        if load_or_dump == 'dump':
            with open(generic_path, 'wb') as file:
                pk.dump(trained_object, file)
        elif load_or_dump == 'load':
            with open(generic_path, 'rb') as file:
                return pk.load(file)

    def load_or_dump_trained_model_CNN(self, trained_model, indicator='TrainedCNN', load_or_dump='dump'):
        """
        Load or save the trained CNN model

        Inputs:
        -------
        - trained_model : The trained CNN model
        - indicator (str): Indicator for the type of trained model
        - load_or_dump (str): 'dump' or 'load'
        """
        json_path = self.create_generic_path(f'{indicator}_json')
        weights_path = self.create_generic_path(f'{indicator}_weights')

        if load_or_dump == 'dump':
            # Serialize model to JSON
            model_json = trained_model.to_json()
            with open(json_path, 'w') as json_file:
                json_file.write(model_json)
            # Save model weights
            trained_model.save_weights(weights_path + '.h5')
        elif load_or_dump == 'load':
            # Load JSON and create model
            with open(json_path, 'r') as json_file:
                loaded_model_json = json_file.read()
            loaded_model = model_from_json(loaded_model_json)
            # Load model weights
            loaded_model.load_weights(weights_path + '.h5')
            return loaded_model
