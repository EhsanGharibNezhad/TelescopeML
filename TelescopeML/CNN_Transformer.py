import tensorflow as tf
from tensorflow import keras 
from keras import layers, models, Model

def build_cnn_transformer_model(
    input_shape,
    num_cnn_filters=64,
    cnn_kernel_size=3,
    cnn_stride=1,
    cnn_padding='same',
    num_transformer_blocks=2,
    num_heads=4,
    transformer_ff_dim=128,
    dropout_rate=0.1,
    output_dim=4
):
    """
    Builds a 1D CNN followed by a Transformer encoder and outputs a vector of specified dimension.

    Parameters:
    - input_shape: Tuple representing the shape of the input (sequence_length, num_features)
    - num_cnn_filters: Number of filters for the CNN layers
    - cnn_kernel_size: Kernel size for the CNN layers
    - cnn_stride: Stride for the CNN layers
    - cnn_padding: Padding type for the CNN layers ('same' or 'valid')
    - num_transformer_blocks: Number of Transformer encoder blocks
    - num_heads: Number of attention heads in the Transformer
    - transformer_ff_dim: Feed-forward network dimension in the Transformer
    - dropout_rate: Dropout rate
    - output_dim: Dimension of the output vector

    Returns:
    - A Keras Model instance
    """

    inputs = layers.Input(shape=input_shape)

    # 1D CNN Layers
    x = layers.Conv1D(filters=num_cnn_filters,
                      kernel_size=cnn_kernel_size,
                      strides=cnn_stride,
                      padding=cnn_padding,
                      activation='relu')(inputs)
    x = layers.BatchNormalization()(x)

    x = layers.Conv1D(filters=num_cnn_filters,
                      kernel_size=cnn_kernel_size,
                      strides=cnn_stride,
                      padding=cnn_padding,
                      activation='relu')(inputs)
    x = layers.BatchNormalization()(x)
    
    x = layers.MaxPooling1D(pool_size=2)(x)

    # You can add more CNN layers if needed
    # Example:
    x = layers.Conv1D(filters=num_cnn_filters * 2,
                      kernel_size=cnn_kernel_size,
                      strides=cnn_stride,
                      padding=cnn_padding,
                      activation='relu')(x)
    x = layers.BatchNormalization()(x)
    
    x = layers.Conv1D(filters=num_cnn_filters * 2,
                      kernel_size=cnn_kernel_size,
                      strides=cnn_stride,
                      padding=cnn_padding,
                      activation='relu')(x)
    x = layers.BatchNormalization()(x)
    
    x = layers.MaxPooling1D(pool_size=2)(x)



    # Prepare for Transformer: Add positional encoding if necessary
    # Here, we'll assume the CNN outputs are suitable for the Transformer

    # Transformer Encoder
    for _ in range(num_transformer_blocks):
        # Multi-Head Self-Attention
        attn_output = layers.MultiHeadAttention(num_heads=num_heads,
                                                key_dim=num_cnn_filters * 2)(x, x)
        attn_output = layers.Dropout(dropout_rate)(attn_output)
        attn_output = layers.LayerNormalization(epsilon=1e-6)(x + attn_output)

        # Feed-Forward Network
        ffn = layers.Dense(transformer_ff_dim, activation='relu')(attn_output)
        ffn = layers.Dense(num_cnn_filters * 2)(ffn)
        ffn = layers.Dropout(dropout_rate)(ffn)
        x = layers.LayerNormalization(epsilon=1e-6)(attn_output + ffn)

    # Global Average Pooling to aggregate the sequence dimension
    x = layers.GlobalAveragePooling1D()(x)

    # Optional: Add more Dense layers
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(dropout_rate)(x)

    x = layers.Dense(32, activation='relu')(x)
    x = layers.Dropout(dropout_rate)(x)

    x = layers.Dense(16, activation='relu')(x)
    x = layers.Dropout(dropout_rate)(x)

    # Output Layer
    outputs = layers.Dense(output_dim, activation='linear')(x)  

    # Build Model
    model = Model(inputs=inputs, outputs=outputs)

    return model

# Example Usage
if __name__ == "__main__":
    # Example input shape: (sequence_length, num_features)
    sequence_length = 104
    num_features = 1  # e.g., univariate time series

    model = build_cnn_transformer_model(
        input_shape=(sequence_length, num_features),
        num_cnn_filters=64,
        cnn_kernel_size=3,
        cnn_stride=1,
        cnn_padding='same',
        num_transformer_blocks=2,
        num_heads=4,
        transformer_ff_dim=128,
        dropout_rate=0.1,
        output_dim=4
    )

    model.compile(optimizer='adam',
                  loss='mse', 
                  metrics=['mae'])

    model.summary()
