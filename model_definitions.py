"""Definitions of keras models."""

from typing import Tuple

import numpy as np
import tensorflow as tf


def define_model_cnn_1_min() -> Tuple[tf.keras.Model, np.ndarray, int, float, int]:
    """Define the structure of the neural network for 1-minute data

    Returns:
        model: keras model
        initial_weights: Array of initial weights used to reset the model to its
            original state
        epochs: Number of epochs
        lr: Learning rate
        bs: Batch size
    """

    inputs = tf.keras.layers.Input((6 * 24 * 7, 13))
    conv1 = tf.keras.layers.Conv1D(50, kernel_size=6, strides=3, activation="relu")(
        inputs
    )
    trim1 = tf.keras.layers.Cropping1D((5, 0))(
        conv1
    )  # crop from left so resulting shape is divisible by 6
    conv2 = tf.keras.layers.Conv1D(50, kernel_size=6, strides=3, activation="relu")(
        trim1
    )
    trim2 = tf.keras.layers.Cropping1D((1, 0))(conv2)
    conv3 = tf.keras.layers.Conv1D(30, kernel_size=6, strides=3, activation="relu")(
        trim2
    )
    trim3 = tf.keras.layers.Cropping1D((5, 0))(conv3)
    conv4 = tf.keras.layers.Conv1D(30, kernel_size=6, strides=3, activation="relu")(
        trim3
    )
    conv5 = tf.keras.layers.Conv1D(30, kernel_size=9, strides=9, activation="relu")(
        conv4
    )
    # extract last data point of previous convolutional layers (left-crop all but one)
    comb1 = tf.keras.layers.Concatenate(axis=2)(
        [
            conv5,
            tf.keras.layers.Cropping1D((334, 0))(conv1),
            tf.keras.layers.Cropping1D((108, 0))(conv2),
            tf.keras.layers.Cropping1D((34, 0))(conv3),
            tf.keras.layers.Cropping1D((8, 0))(conv4),
        ]
    )
    dense = tf.keras.layers.Dense(50, activation="relu")(comb1)
    output = tf.keras.layers.Flatten()(tf.keras.layers.Dense(1)(dense))
    model = tf.keras.Model(inputs, output)
    initial_weights = model.get_weights()
    epochs = 2
    lr = 0.00025
    bs = 32
    return model, initial_weights, epochs, lr, bs


def define_model_cnn_hourly() -> Tuple[tf.keras.Model, np.ndarray, int, float, int]:
    """Define the structure of the neural network for hourly data.

    Returns:
        model: keras model
        initial_weights: Array of initial weights used to reset the model to its
            original state
        epochs: Number of epochs
        lr: Learning rate
        bs: Batch size
    """

    inputs = tf.keras.layers.Input((24 * 7, 6))
    conv1 = tf.keras.layers.Conv1D(50, kernel_size=1, strides=1, activation="relu")(
        inputs
    )
    conv2 = tf.keras.layers.Conv1D(50, kernel_size=6, strides=3, activation="relu")(
        conv1
    )
    trim2 = tf.keras.layers.Cropping1D((1, 0))(conv2)
    conv3 = tf.keras.layers.Conv1D(30, kernel_size=6, strides=3, activation="relu")(
        trim2
    )
    trim3 = tf.keras.layers.Cropping1D((5, 0))(conv3)
    conv4 = tf.keras.layers.Conv1D(30, kernel_size=6, strides=3, activation="relu")(
        trim3
    )
    conv5 = tf.keras.layers.Conv1D(30, kernel_size=3, strides=3, activation="relu")(
        conv4
    )
    # extract last data point of previous convolutional layers (left-crop all but one)
    comb1 = tf.keras.layers.Concatenate(axis=2)(
        [
            conv5,
            tf.keras.layers.Cropping1D((334, 0))(conv1),
            tf.keras.layers.Cropping1D((108, 0))(conv2),
            tf.keras.layers.Cropping1D((34, 0))(conv3),
            tf.keras.layers.Cropping1D((8, 0))(conv4),
        ]
    )
    dense = tf.keras.layers.Dense(50, activation="relu")(comb1)
    output = tf.keras.layers.Flatten()(tf.keras.layers.Dense(1)(dense))
    model = tf.keras.Model(inputs, output)
    initial_weights = model.get_weights()
    epochs = 2
    lr = 0.00025
    bs = 32
    return model, initial_weights, epochs, lr, bs


def define_model_cnn_lstm() -> Tuple[tf.keras.Model, np.ndarray, int, float, int]:
    """Define the structure of the neural network.

    Returns:
        model: keras model
        initial_weights: Array of initial weights used to reset the model to its
            original state
        epochs: Number of epochs
        lr: Learning rate
        bs: Batch size
    """

    inputs = tf.keras.layers.Input((6 * 24 * 7, 13))
    conv1 = tf.keras.layers.Conv1D(50, kernel_size=6, strides=3, activation="relu")(
        inputs
    )
    lstm1 = tf.keras.layers.LSTM(50, return_sequences=False)(conv1)
    trim1 = tf.keras.layers.Cropping1D((5, 0))(
        conv1
    )  # crop from left so resulting shape is divisible by 6
    conv2 = tf.keras.layers.Conv1D(50, kernel_size=6, strides=3, activation="relu")(
        trim1
    )
    lstm2 = tf.keras.layers.LSTM(50, return_sequences=False)(conv2)
    trim2 = tf.keras.layers.Cropping1D((1, 0))(conv2)
    conv3 = tf.keras.layers.Conv1D(30, kernel_size=6, strides=3, activation="relu")(
        trim2
    )
    lstm3 = tf.keras.layers.LSTM(50, return_sequences=False)(conv3)
    trim3 = tf.keras.layers.Cropping1D((5, 0))(conv3)
    conv4 = tf.keras.layers.Conv1D(30, kernel_size=6, strides=3, activation="relu")(
        trim3
    )
    lstm4 = tf.keras.layers.LSTM(30, return_sequences=False)(conv4)
    conv5 = tf.keras.layers.Conv1D(30, kernel_size=9, strides=9, activation="relu")(
        conv4
    )
    # extract last data point of previous convolutional layers (left-crop all but one)
    comb1 = tf.keras.layers.Concatenate(axis=2)(
        [
            conv5, tf.expand_dims(lstm1, -1), tf.expand_dims(lstm2, -1), tf.expand_dims(lstm3, -1), tf.expand_dims(lstm4, -1)
        ]
    )
    dense = tf.keras.layers.Dense(50, activation="relu")(comb1)
    output = tf.keras.layers.Flatten()(tf.keras.layers.Dense(1)(dense))
    model = tf.keras.Model(inputs, output)
    initial_weights = model.get_weights()
    epochs = 2
    lr = 0.00025
    bs = 32
    return model, initial_weights, epochs, lr, bs


def define_model_lstm() -> Tuple[tf.keras.Model, np.ndarray, int, float, int]:
    """Define the structure of the neural network.

    Returns:
        model: keras model
        initial_weights: Array of initial weights used to reset the model to its
            original state
        epochs: Number of epochs
        lr: Learning rate
        bs: Batch size
    """

    inputs = tf.keras.layers.Input((6 * 24 * 7, 13))
    lstm1 = tf.keras.layers.LSTM(50, activation="tanh", return_sequences=True)(inputs)
    lstm2 = tf.keras.layers.LSTM(50, activation="tanh", return_sequences=True)(lstm1)
    lstm3 = tf.keras.layers.LSTM(50, activation="tanh", return_sequences=False)(lstm2)
    dense = tf.keras.layers.Dense(50, activation="tanh")(lstm3)
    output = tf.keras.layers.Dense(1)(dense)
    model = tf.keras.Model(inputs, output)
    initial_weights = model.get_weights()
    epochs = 5
    lr = 0.00025
    bs = 32
    return model, initial_weights, epochs, lr, bs


def define_model_transformer() -> Tuple[tf.keras.Model, np.ndarray, int, float, int]:
    """Define the structure of the neural network.

    This code is adapted from a tutorial on the keras website by Theodoros Ntakouris:
    https://keras.io/examples/timeseries/timeseries_transformer_classification/

    Returns:
        model: keras model
        initial_weights: Array of initial weights used to reset the model to its
            original state
        epochs: Number of epochs
        lr: Learning rate
        bs: Batch size
    """

    def transformer_encoder(inputs: tf.Tensor, head_size: int, num_heads: int,
                            ff_dim: int, dropout: float) -> tf.Tensor:
        # Normalization and Attention
        x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(inputs)
        x = tf.keras.layers.MultiHeadAttention(
            key_dim=head_size, num_heads=num_heads, dropout=dropout
        )(x, x)
        x = tf.keras.layers.Dropout(dropout)(x)
        res = x + inputs

        # Feed Forward Part
        x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(res)
        x = tf.keras.layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(x)
        x = tf.keras.layers.Dropout(dropout)(x)
        x = tf.keras.layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
        return x + res

    num_transformer_blocks = 2  # how many consecutive transformer layers
    head_size = 50  # channels in the attention head
    inputs = tf.keras.Input((6 * 24 * 7, 13))
    x = inputs
    num_heads = 2
    ff_dim = 50
    dropout = 0
    mlp_units = [20]
    mlp_dropout = 0
    for _ in range(num_transformer_blocks):
        x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout)

    x = tf.keras.layers.GlobalAveragePooling1D(data_format="channels_first")(x)
    for dim in mlp_units:
        x = tf.keras.layers.Dense(dim, activation="relu")(x)
        x = tf.keras.layers.Dropout(mlp_dropout)(x)
    output = tf.keras.layers.Dense(1, activation="softmax")(x)
    model = tf.keras.Model(inputs, output)
    initial_weights = model.get_weights()
    epochs = 5
    lr = 0.001
    bs = 32
    return model, initial_weights, epochs, lr, bs
