# Import functions from other modules ============================
from IO_utils import LoadSave

# Import python libraries ========================================
# Dataset manipulation libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.python.keras.layers import Input, Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from tensorflow.python.keras.models import Model
from sklearn.model_selection import train_test_split

# Generate synthetic 1D signal data
def generate_synthetic_data(num_samples=1000, sequence_length=128):
    X = np.random.randn(num_samples, sequence_length, 1)  
    y = np.random.randint(0, 2, size=(num_samples, 1)) 
    return X, y

# Preprocess data: normalize and split
def preprocess_data(X, y):

    # Standardizing the data
    X = (X - np.mean(X)) / np.std(X)
    
    # Splitting data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

# Define the 1D CNN model
def build_model(sequence_length):
    inputs = Input(shape=(sequence_length, 1))
    
    # CNN Block
    x = Conv1D(32, kernel_size=3, activation='relu', padding='same')(inputs)
    x = MaxPooling1D(pool_size=2)(x)
    x = Conv1D(64, kernel_size=3, activation='relu', padding='same')(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.5)(x)
    outputs = Dense(1, activation='sigmoid')(x) 

    model = Model(inputs=inputs, outputs=outputs)
    return model

if __name__ == "__main__":
    sequence_length = 128  

    # Generate synthetic data
    X, y = generate_synthetic_data(num_samples=1000, sequence_length=sequence_length)
    
    # Preprocess data
    X_train, X_test, y_train, y_test = preprocess_data(X, y)
    
    print(f"X_train type: {type(X_train)}, shape: {X_train.shape}")
    print(f"y_train type: {type(y_train)}, shape: {y_train.shape}")

    # Build and compile the model
    model = build_model(sequence_length)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    model.summary()
    
    # Train the model
    try:
        history = model.fit(X_train, y_train, epochs=20, batch_size=32)
    except Exception as e:
        print(f"An error occurred during model training: {e}")
        history = None
    
    # Evaluating model
    if history:
        try:
            test_loss, test_acc = model.evaluate(X_test, y_test)
            print(f'Test Accuracy: {test_acc:.4f}')
        except Exception as e:
            print(f"An error occurred during model evaluation: {e}")
    
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'])
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend(['Train'])
        
        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'])
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend(['Train'])
        
        plt.tight_layout()
        plt.show()
    
    # Initialize LoadSave instance
    ml_model_str = '1d_cnn_model'
    is_feature_improved = True
    is_augmented = False
    is_tuned = True

    load_save = LoadSave(ml_model_str, is_feature_improved, is_augmented, is_tuned)
    
    # Saving the model using LoadSave for future use
    try:
        load_save.load_or_dump_trained_object(
            trained_object=model,
            output_indicator='TrainedModel', 
            load_or_dump='dump'
        )
    except Exception as e:
        print(f"An error occurred during model saving: {e}")
