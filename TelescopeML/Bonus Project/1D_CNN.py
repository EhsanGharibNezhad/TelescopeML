import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import tensorflow as tf
from keras._tf_keras.keras.layers import Input, Conv1D, MaxPooling1D, Dense, Dropout, BatchNormalization, MultiHeadAttention, LayerNormalization, GlobalAveragePooling1D
from keras._tf_keras.keras.models import Model
from keras._tf_keras.keras.optimizers import Adam
from keras._tf_keras.keras.losses import MeanSquaredError
from keras._tf_keras.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from keras._tf_keras.keras.regularizers import l2

# Step 1: Define the 1D CNN model
def build_cnn(input_shape):
    inputs = Input(shape=input_shape)
    
    # 1D Convolutional Layers
    x = Conv1D(filters=64, kernel_size=3, activation='relu', padding='same', kernel_regularizer=l2(0.01))(inputs)
    print(f"Shape after Conv1D 64 filters: {x.shape}")
    x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=2)(x)

    x = Conv1D(filters=128, kernel_size=3, activation='relu', padding='same', kernel_regularizer=l2(0.01))(x)
    print(f"Shape after Conv1D 128 filters: {x.shape}")
    x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=2)(x)
    
    x = Conv1D(filters=256, kernel_size=3, activation='relu', padding='same', kernel_regularizer=l2(0.01))(x)
    print(f"Shape after Conv1D 256 filters: {x.shape}")
    x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=2)(x)

    # Do not apply Global Average Pooling here to preserve the sequence length
    # Add a Dropout layer to reduce overfitting
    x = Dropout(0.6)(x)

    return inputs, x

# Step 2: Transformer Block
def transformer_block(x, num_heads, ff_dim, rate=0.1):
    # Multi-Head Attention layer
    attention_output = MultiHeadAttention(num_heads=num_heads, key_dim=ff_dim)(x, x)
    attention_output = Dropout(rate)(attention_output)
    attention_output = LayerNormalization(epsilon=1e-6)(attention_output + x)

    # Feed Forward Network
    ffn_output = Dense(ff_dim, activation="relu")(attention_output)
    ffn_output = Dense(x.shape[-1])(ffn_output)
    ffn_output = Dropout(rate)(ffn_output)
    return LayerNormalization(epsilon=1e-6)(ffn_output + attention_output)

# Step 3: Build the entire model
def build_model(input_shape, num_heads, ff_dim, num_outputs):
    inputs, cnn_features = build_cnn(input_shape)
    
    # Apply transformer layers on the CNN features
    transformer_features = transformer_block(cnn_features, num_heads=num_heads, ff_dim=ff_dim)
    
    # Global average pooling to reduce the sequence length after Transformer
    transformer_features = GlobalAveragePooling1D()(transformer_features)
    
    # Final output layer
    outputs = Dense(num_outputs, activation='linear')(transformer_features)

    # Build and return the full model
    model = Model(inputs, outputs)
    return model

# Step 4: Compile the model
def compile_model(model):
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss=MeanSquaredError(),
        metrics=['mae']
    )

# Step 5: Prepare Data (Example data structure - replace with actual data)
def prepare_data():
    # Step 1: Load the dataset
    path = os.path.join('..', 'browndwarf_R100_v4_newWL_v3.csv.bz2')
    data = pd.read_csv(path, compression='bz2')

    #Take log of temperature before standardization
    data['temperature'] = np.log10(data['temperature'])

    # Step 2: Separate outputs and inputs
    outputs = data.iloc[:, :4]  # First 4 columns (outputs)
    inputs = data.iloc[:, 4:4+104]  # Next 104 columns (inputs)

    # Step 3: Normalize or standardize the input data
    # Standardize (mean = 0, std = 1)
    scaler = StandardScaler()
    standardized_inputs = scaler.fit_transform(inputs)

    # Step 4: Split the data into training, validation, and test sets
    X_train, X_temp, y_train, y_temp = train_test_split(standardized_inputs, outputs, test_size=0.30, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42)
    
    # Reshape input to be 3D for Conv1D (batch_size, sequence_length, features)
    X_train = np.expand_dims(X_train, axis=-1)
    X_val = np.expand_dims(X_val, axis=-1)
    X_test = np.expand_dims(X_test, axis=-1)
    
    return X_train, X_val, X_test, y_train, y_val, y_test

# Step 6: Train the model
def train_model(model, X_train, X_val, y_train, y_val, epochs=100, batch_size=32):
    # Define the learning rate scheduler callback
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',   # Metric to monitor
        factor=0.5,           # Factor by which learning rate will be reduced
        patience=5,           # Number of epochs with no improvement after which learning rate will be reduced
        min_lr=0.00001        # Minimum learning rate threshold
    )

    # Define the early stopping callback
    early_stopping = EarlyStopping(
        monitor='val_loss',   # Metric to monitor
        patience=10,           # Stop after 5 epochs with no improvement
        restore_best_weights=True  # Restore the weights of the best epoch
    )

    # Define a ModelCheckpoint callback to save the best model
    model_checkpoint = ModelCheckpoint(
        filepath='/Users/Abhi/Documents/GitHub/TelescopeML_project/TelescopeML/TelescopeML/Bonus Project/best_model.keras',       # Filepath to save the model
        monitor='val_loss',   # Metric to monitor
        save_best_only=True,  # Save only the best model
        mode='min'             # Mode to determine the best model
    )
    
    # Train the model with the learning rate scheduler and early stopping
    currModel = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[reduce_lr, early_stopping, model_checkpoint]  # Add both learning rate scheduler and early stopping
    )
    return currModel

# Step 7: Evaluate the model
def evaluate_model(model, X_test, y_test):
    results = model.evaluate(X_test, y_test)
    print(f"Test Loss: {results[0]}, Test MAE: {results[1]}")

# Step 8: Execute the entire pipeline
def run_pipeline():
    # Prepare data
    X_train, X_val, X_test, y_train, y_val, y_test = prepare_data()

    # Build the model (set appropriate hyperparameters)
    input_shape = (104, 1)  # Input is a sequence of 104 flux values
    num_heads = 4           # Number of attention heads for the transformer
    ff_dim = 64            # Feed-forward network size inside the transformer
    num_outputs = 4         # The output will have 4 values (for regression task)

    # Build and compile the model
    model = build_model(input_shape, num_heads, ff_dim, num_outputs)
    compile_model(model)

    # Train the model
    history = train_model(model, X_train, X_val, y_train, y_val, epochs=100, batch_size=32)

    # Evaluate the model
    evaluate_model(model, X_test, y_test)

# Run the pipeline
run_pipeline()
