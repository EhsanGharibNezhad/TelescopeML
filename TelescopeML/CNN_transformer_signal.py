import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, Input, MultiHeadAttention, LayerNormalization, Add


class CNNTransformerModel:
    def __init__(self, length, num_signals):
        self.length = length
        self.num_signals = num_signals
        self.signals = self.generate_synthetic_signal()
        self.labels = np.random.randint(0, 2, num_signals)

    def generate_synthetic_signal(self):
        signals = np.sin(np.linspace(0, 10, self.length)) + np.random.normal(0, 0.5, (self.num_signals, self.length))
        return np.array(signals)

    def preprocess_data(self, test_size=0.3, val_size=0.5):
        # Normalize the signals
        scaler = StandardScaler()
        signals = scaler.fit_transform(self.signals)
        
        # Reshape data for the CNN 
        signals = signals[..., np.newaxis]
        
        # Split the data into training, validation, and test sets
        X_train, X_temp, y_train, y_temp = train_test_split(signals, self.labels, test_size=test_size, random_state=42)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=val_size, random_state=42)
        
        return X_train, X_val, X_test, y_train, y_val, y_test

    def build_cnn_transformer(self, input_shape):
        inputs = Input(shape=input_shape)
        
        # CNN layers
        x = Conv1D(filters=32, kernel_size=3, activation='relu')(inputs)
        x = MaxPooling1D(pool_size=2)(x)
        x = Dropout(0.5)(x)
        x = Conv1D(filters=64, kernel_size=3, activation='relu')(x)
        x = MaxPooling1D(pool_size=2)(x)
        x = Dropout(0.5)(x)
        
        # Transformer layers
        x = Flatten()(x)
        x = Dense(128, activation='relu')(x)
        x = tf.expand_dims(x, axis=1) 
        attn_output = MultiHeadAttention(num_heads=8, key_dim=128)(x, x)
        attn_output = Add()([x, attn_output])
        attn_output = LayerNormalization()(attn_output)
        
        x = Flatten()(attn_output)
        x = Dense(100, activation='relu')(x)
        outputs = Dense(1, activation='sigmoid')(x) 

        model = Model(inputs, outputs)
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model

    def train_model(self, model, X_train, y_train, X_val, y_val, epochs=20):
        history = model.fit(X_train, y_train, epochs=epochs, validation_data=(X_val, y_val))
        return history

    def evaluate_model(self, model, X_test, y_test):
        test_loss, test_accuracy = model.evaluate(X_test, y_test)
        print(f"Test Loss: {test_loss}")
        print(f"Test Accuracy: {test_accuracy}")
        return test_loss, test_accuracy

    def save_model(self, model, filename):
        model.save(filename)
        print(f"Model saved to {filename}")

