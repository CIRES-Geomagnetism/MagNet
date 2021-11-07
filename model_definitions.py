"""Definitions of keras models."""

from typing import List, Tuple

import numpy as np
import tensorflow as tf


def define_model_cnn_1_min() -> Tuple[
    tf.keras.Model, List[np.ndarray], int, float, int
]:
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
    epochs = 3
    lr = 0.00025
    bs = 32
    return model, initial_weights, epochs, lr, bs


def define_model_cnn_hourly() -> Tuple[
    tf.keras.Model, List[np.ndarray], int, float, int
]:
    """Define the structure of the neural network for hourly data.

    Returns:
        model: keras model
        initial_weights: Array of initial weights used to reset the model to its
            original state
        epochs: Number of epochs
        lr: Learning rate
        bs: Batch size
    """

    inputs = tf.keras.layers.Input((24 * 7, 7))
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
            tf.keras.layers.Cropping1D((167, 0))(conv1),
            tf.keras.layers.Cropping1D((54, 0))(conv2),
            tf.keras.layers.Cropping1D((16, 0))(conv3),
            tf.keras.layers.Cropping1D((2, 0))(conv4),
        ]
    )
    dense = tf.keras.layers.Dense(50, activation="relu")(comb1)
    output = tf.keras.layers.Flatten()(tf.keras.layers.Dense(1)(dense))
    model = tf.keras.Model(inputs, output)
    initial_weights = model.get_weights()
    epochs = 8
    lr = 0.00025
    bs = 32
    return model, initial_weights, epochs, lr, bs


def define_model_cnn_hybrid() -> Tuple[
    Tuple[tf.keras.Model, List[np.ndarray], int, float, int],
    Tuple[tf.keras.Model, List[np.ndarray], int, float, int],
]:
    """
    Hybrid CNN model with different sets of layers for hourly and 1-minute data.

    """

    inputs = tf.keras.layers.Input((24 * 7 * 6, 13))
    # hourly part
    hour_padded_inputs = inputs[:, :, :7]
    # dim 1 has length 24 * 6 * 7, but values are repeated in blocks of 6, so use
    # average pooling to condense to size 24 * 7
    hourly_avg = tf.keras.layers.AveragePooling1D(pool_size=6, strides=6)(
        hour_padded_inputs
    )
    hourly_conv1 = tf.keras.layers.Conv1D(
        50, kernel_size=1, strides=1, activation="relu", name="hourly_conv1"
    )(hourly_avg)
    hourly_conv2 = tf.keras.layers.Conv1D(
        50, kernel_size=6, strides=3, activation="relu", name="hourly_conv2"
    )(hourly_conv1)
    hourly_trim1 = tf.keras.layers.Cropping1D((1, 0), name="hourly_trim1")(hourly_conv2)
    hourly_conv3 = tf.keras.layers.Conv1D(
        30, kernel_size=6, strides=3, activation="relu", name="hourly_conv3"
    )(hourly_trim1)
    hourly_trim2 = tf.keras.layers.Cropping1D((2, 0), name="hourly_trim2")(hourly_conv3)

    # high-frequency part
    minute_conv1 = tf.keras.layers.Conv1D(
        50,
        kernel_size=6,
        strides=1,
        activation="relu",
        padding="causal",
        name="minute_conv1",
    )(inputs)
    # minute_conv2 has output size 168 = 24 * 7, so each output represents an hour
    minute_conv2 = tf.keras.layers.Conv1D(
        50,
        kernel_size=6,
        strides=6,
        activation="relu",
        padding="causal",
        name="minute_conv2",
    )(minute_conv1)
    # the layers of the high-frequency part have same structure as hourly model, and we
    # concatenate their outputs with the outputs of the hourly model
    minute_concat1 = tf.keras.layers.Concatenate(name="minute_concat1")(
        [minute_conv2, hourly_conv1]
    )
    minute_conv3 = tf.keras.layers.Conv1D(
        50, kernel_size=6, strides=3, activation="relu", name="minute_conv3"
    )(minute_concat1)
    minute_trim1 = tf.keras.layers.Cropping1D((1, 0), name="minute_trim1")(minute_conv3)
    minute_concat2 = tf.keras.layers.Concatenate(name="minute_concat2")(
        [minute_trim1, hourly_trim1]
    )
    minute_conv4 = tf.keras.layers.Conv1D(
        50, kernel_size=6, strides=3, activation="relu", name="minute_conv4"
    )(minute_concat2)
    minute_trim2 = tf.keras.layers.Cropping1D((2, 0), name="minute_trim2")(minute_conv4)
    minute_concat3 = tf.keras.layers.Concatenate(name="minute_concat3")(
        [minute_trim2, hourly_trim2]
    )
    minute_conv5 = tf.keras.layers.Conv1D(
        30, kernel_size=6, strides=3, activation="relu", name="minute_conv5"
    )(minute_concat3)
    minute_conv6 = tf.keras.layers.Conv1D(
        30, kernel_size=4, strides=4, activation="relu", name="minute_conv6"
    )(minute_conv5)
    # extract last data point of previous convolutional layers (left-crop all but one)
    minute_comb1 = tf.keras.layers.Concatenate(axis=2, name="minute_comb1")(
        [
            minute_conv6,
            tf.keras.layers.Cropping1D((167, 0))(minute_conv2),
            tf.keras.layers.Cropping1D((54, 0))(minute_conv3),
            tf.keras.layers.Cropping1D((16, 0))(minute_conv4),
            tf.keras.layers.Cropping1D((3, 0))(minute_conv5),
        ]
    )
    minute_dense = tf.keras.layers.Dense(50, activation="relu", name="minute_dense")(
        minute_comb1
    )
    output = tf.keras.layers.Flatten()(tf.keras.layers.Dense(1)(minute_dense))
    minute_model = tf.keras.Model(inputs, output)
    minute_weights = minute_model.get_weights()

    # create the hourly model
    hour_input = tf.keras.layers.Input((7 * 24, 7))
    # add the convolutions used in the 1-minute model
    x = hour_input
    # dictionary of layer outputs (can't refer to outputs from minute model because
    # they are created from different initial input)
    tensor_dict = {}
    for layer in minute_model.layers:
        if "hour" in layer.name:
            # clone the layers, code from here:
            # https://github.com/keras-team/keras/issues/13140#issuecomment-766394677
            config = layer.get_config()
            tensor_dict[layer.name] = type(layer).from_config(config)(x)
            x = tensor_dict[layer.name]
    # add some more convolutions
    hourly_conv4 = tf.keras.layers.Conv1D(
        30, kernel_size=6, strides=3, activation="relu", name="hourly_conv4"
    )(tensor_dict["hourly_trim2"])
    hourly_conv5 = tf.keras.layers.Conv1D(
        30, kernel_size=3, strides=3, activation="relu", name="hourly_conv5"
    )(hourly_conv4)
    # extract last data point of previous convolutional layers (left-crop all but one)
    hourly_comb1 = tf.keras.layers.Concatenate(axis=2)(
        [
            hourly_conv5,
            tf.keras.layers.Cropping1D((167, 0))(tensor_dict["hourly_conv1"]),
            tf.keras.layers.Cropping1D((54, 0))(tensor_dict["hourly_conv2"]),
            tf.keras.layers.Cropping1D((16, 0))(tensor_dict["hourly_conv3"]),
            tf.keras.layers.Cropping1D((3, 0))(hourly_conv4),
        ]
    )
    hourly_dense = tf.keras.layers.Dense(50, activation="relu", name="hour8")(
        hourly_comb1
    )
    hour_output = tf.keras.layers.Flatten()(tf.keras.layers.Dense(1)(hourly_dense))
    hour_model = tf.keras.Model(hour_input, hour_output)
    hour_weights = hour_model.get_weights()
    epochs = 1
    lr = 0.00025
    bs = 32
    return (
        (minute_model, minute_weights, epochs, lr, bs),
        (hour_model, hour_weights, epochs, lr, bs),
    )


def define_model_cnn_lstm_1_min() -> Tuple[
    tf.keras.Model, List[np.ndarray], int, float, int
]:
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
    # extract output of previous lstm layers
    comb1 = tf.keras.layers.Concatenate()(
        [tf.keras.layers.Flatten()(conv5), lstm1, lstm2, lstm3, lstm4]
    )
    dense = tf.keras.layers.Dense(50, activation="relu")(comb1)
    output = tf.keras.layers.Flatten()(tf.keras.layers.Dense(1)(dense))
    model = tf.keras.Model(inputs, output)
    initial_weights = model.get_weights()
    epochs = 2
    lr = 0.00025
    bs = 32
    return model, initial_weights, epochs, lr, bs


def define_model_cnn_lstm_hourly() -> Tuple[
    tf.keras.Model, List[np.ndarray], int, float, int
]:
    """Define the structure of the neural network.

    Returns:
        model: keras model
        initial_weights: Array of initial weights used to reset the model to its
            original state
        epochs: Number of epochs
        lr: Learning rate
        bs: Batch size
    """

    inputs = tf.keras.layers.Input((24 * 7, 7))
    conv1 = tf.keras.layers.Conv1D(50, kernel_size=1, strides=1, activation="relu")(
        inputs
    )
    lstm1 = tf.keras.layers.LSTM(50, return_sequences=False)(conv1)
    conv2 = tf.keras.layers.Conv1D(50, kernel_size=6, strides=3, activation="relu")(
        conv1
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
    conv5 = tf.keras.layers.Conv1D(30, kernel_size=3, strides=3, activation="relu")(
        conv4
    )
    #  extract output of previous lstm layers
    comb1 = tf.keras.layers.Concatenate()(
        [tf.keras.layers.Flatten()(conv5), lstm1, lstm2, lstm3, lstm4]
    )
    dense = tf.keras.layers.Dense(50, activation="relu")(comb1)
    output = tf.keras.layers.Flatten()(tf.keras.layers.Dense(1)(dense))
    model = tf.keras.Model(inputs, output)
    initial_weights = model.get_weights()
    epochs = 3
    lr = 0.00025
    bs = 32
    return model, initial_weights, epochs, lr, bs


def define_model_lstm_1_min() -> Tuple[
    tf.keras.Model, List[np.ndarray], int, float, int
]:
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
    conv1 = tf.keras.layers.Conv1D(50, kernel_size=6, strides=1, padding="same")(inputs)
    lstm1 = tf.keras.layers.LSTM(50, activation="tanh", return_sequences=True)(conv1)
    lstm2 = tf.keras.layers.LSTM(50, activation="tanh", return_sequences=True)(lstm1)
    lstm3 = tf.keras.layers.LSTM(50, activation="tanh", return_sequences=False)(lstm2)
    dense = tf.keras.layers.Dense(50, activation="tanh")(lstm3)
    output = tf.keras.layers.Dense(1)(dense)
    model = tf.keras.Model(inputs, output)
    initial_weights = model.get_weights()
    epochs = 7
    lr = 0.00025
    bs = 32
    return model, initial_weights, epochs, lr, bs


def define_model_lstm_hourly() -> Tuple[
    tf.keras.Model, List[np.ndarray], int, float, int
]:
    """Define the structure of the neural network.

    Returns:
        model: keras model
        initial_weights: Array of initial weights used to reset the model to its
            original state
        epochs: Number of epochs
        lr: Learning rate
        bs: Batch size
    """

    inputs = tf.keras.layers.Input((24 * 7, 7))
    conv1 = tf.keras.layers.Conv1D(50, kernel_size=6, strides=1, padding="same")(inputs)
    lstm1 = tf.keras.layers.LSTM(50, activation="tanh", return_sequences=True)(conv1)
    lstm2 = tf.keras.layers.LSTM(50, activation="tanh", return_sequences=True)(lstm1)
    lstm3 = tf.keras.layers.LSTM(50, activation="tanh", return_sequences=False)(lstm2)
    dense = tf.keras.layers.Dense(50, activation="tanh")(lstm3)
    output = tf.keras.layers.Dense(1)(dense)
    model = tf.keras.Model(inputs, output)
    initial_weights = model.get_weights()
    epochs = 10
    lr = 0.00025
    bs = 32
    return model, initial_weights, epochs, lr, bs


def define_model_transformer() -> Tuple[
    tf.keras.Model, List[np.ndarray], int, float, int
]:
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

    def transformer_encoder(
        inputs: tf.Tensor, head_size: int, num_heads: int, ff_dim: int, dropout: float
    ) -> tf.Tensor:
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
