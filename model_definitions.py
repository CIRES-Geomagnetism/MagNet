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
    # dim 1 has length 24 * 6 * 7, so use average pooling to condense to size 24 * 7
    # in the result, the value for each hour is the average of the minutes of the
    # previous hour. This is equivalent to what's done in
    # preprocessing.combine_old_and_new_data
    hourly_avg = tf.keras.layers.AveragePooling1D(pool_size=6, strides=6)(
        hour_padded_inputs
    )
    hourly_conv1 = tf.keras.layers.Conv1D(
        32, kernel_size=1, strides=1, activation="relu", name="hourly_conv1"
    )(hourly_avg)
    hourly_conv2 = tf.keras.layers.Conv1D(
        64, kernel_size=6, strides=3, activation="relu", name="hourly_conv2"
    )(hourly_conv1)
    hourly_trim1 = tf.keras.layers.Cropping1D((1, 0), name="hourly_trim1")(hourly_conv2)
    hourly_conv3 = tf.keras.layers.Conv1D(
        64, kernel_size=6, strides=3, activation="relu", name="hourly_conv3"
    )(hourly_trim1)
    hourly_trim2 = tf.keras.layers.Cropping1D((5, 0), name="hourly_trim2")(hourly_conv3)
    hourly_conv4 = tf.keras.layers.Conv1D(
        32, kernel_size=6, strides=3, activation="relu", name="hourly_conv4"
    )(hourly_trim2)
    hourly_conv5 = tf.keras.layers.Conv1D(
        32, kernel_size=3, strides=3, activation="relu", name="hourly_conv5"
    )(hourly_conv4)
    hourly_comb1 = tf.keras.layers.Concatenate(axis=2)(
        [
            hourly_conv5,
            tf.keras.layers.Cropping1D((167, 0))(hourly_conv1),
            tf.keras.layers.Cropping1D((54, 0))(hourly_conv2),
            tf.keras.layers.Cropping1D((16, 0))(hourly_conv3),
            tf.keras.layers.Cropping1D((2, 0))(hourly_conv4),
        ]
    )
    hourly_dense = tf.keras.layers.Dense(64, activation="relu", name="hour8")(
        hourly_comb1
    )

    # high-frequency part
    minute_conv1 = tf.keras.layers.Conv1D(
        128,
        kernel_size=6,
        strides=3,
        activation="relu",
        padding="causal",
        name="minute_conv1",
    )(inputs)
    # minute_conv2 has output size 168 = 24 * 7, so each output represents an hour
    minute_conv2 = tf.keras.layers.Conv1D(
        64,
        kernel_size=6,
        strides=2,
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
        64, kernel_size=6, strides=3, activation="relu", name="minute_conv3"
    )(minute_concat1)
    minute_trim1 = tf.keras.layers.Cropping1D((1, 0), name="minute_trim1")(minute_conv3)
    minute_concat2 = tf.keras.layers.Concatenate(name="minute_concat2")(
        [minute_trim1, hourly_trim1]
    )
    minute_conv4 = tf.keras.layers.Conv1D(
       64, kernel_size=6, strides=3, activation="relu", name="minute_conv4"
    )(minute_concat2)
    minute_trim2 = tf.keras.layers.Cropping1D((5, 0), name="minute_trim2")(minute_conv4)
    minute_concat3 = tf.keras.layers.Concatenate(name="minute_concat3")(
        [minute_trim2, hourly_trim2]
    )
    minute_conv5 = tf.keras.layers.Conv1D(
        32, kernel_size=6, strides=3, activation="relu", name="minute_conv5"
    )(minute_concat3)
    minute_concat4 = tf.keras.layers.Concatenate(name="minute_concat4")([minute_conv5, hourly_conv4])
    minute_conv6 = tf.keras.layers.Conv1D(
        32, kernel_size=3, strides=3, activation="relu", name="minute_conv6"
    )(minute_concat4)
    # extract last data point of previous convolutional layers (left-crop all but one)
    minute_comb1 = tf.keras.layers.Concatenate(axis=2, name="minute_comb1")(
        [
            minute_conv6,
            hourly_conv5,
            tf.keras.layers.Cropping1D((335, 0))(minute_conv1),
            tf.keras.layers.Cropping1D((167, 0))(minute_conv2),
            tf.keras.layers.Cropping1D((54, 0))(minute_conv3),
            tf.keras.layers.Cropping1D((16, 0))(minute_conv4),
            tf.keras.layers.Cropping1D((2, 0))(minute_conv5),
        ]
    )
    minute_dense = tf.keras.layers.Dense(256, activation="relu", name="minute_dense")(
        minute_comb1
    )
    comb_dense = tf.keras.layers.Dense(256, activation="relu")(
        tf.keras.layers.Concatenate()([minute_dense, hourly_dense])
    )
    output = tf.keras.layers.Flatten()(tf.keras.layers.Dense(1)(comb_dense))
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
    # add final dense layer
    # extract last data point of previous convolutional layers (left-crop all but one)
    hourly_comb1 = tf.keras.layers.Concatenate(axis=2)(
        [
            tensor_dict["hourly_conv5"],
            tf.keras.layers.Cropping1D((167, 0))(tensor_dict["hourly_conv1"]),
            tf.keras.layers.Cropping1D((54, 0))(tensor_dict["hourly_conv2"]),
            tf.keras.layers.Cropping1D((16, 0))(tensor_dict["hourly_conv3"]),
            tf.keras.layers.Cropping1D((2, 0))(tensor_dict["hourly_conv4"]),
        ]
    )
    hourly_dense = tf.keras.layers.Dense(64, activation="relu")(
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

    """LSTM model, similar to first-place model:
    https://github.com/drivendataorg/magnet-geomagnetic-field/blob/399c123f1470c0f4de5c2a27122e9954497190ac/1st_Place/MagNet_Model_the_Geomagnetic_Field_first_place_solution.ipynb"""
    input1 = tf.keras.layers.Input((24 * 7 * 6, 13))
    lstm1 = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, return_sequences=True))(input1)
    gru1 = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(256, return_sequences=True))(lstm1)
    flatten = tf.keras.layers.Flatten()(gru1)
    dense = tf.keras.layers.Dense(1)(flatten)
    model = tf.keras.models.Model(inputs=input1, outputs=dense)

    initial_weights = model.get_weights()
    epochs = 7
    lr = 0.00025
    bs = 32
    return model, initial_weights, epochs, lr, bs



def define_model_lstm_hourly() -> Tuple[
    tf.keras.Model, List[np.ndarray], int, float, int
]:

    """LSTM model, similar to first-place model:
    https://github.com/drivendataorg/magnet-geomagnetic-field/blob/399c123f1470c0f4de5c2a27122e9954497190ac/1st_Place/MagNet_Model_the_Geomagnetic_Field_first_place_solution.ipynb"""
    input1 = tf.keras.layers.Input((24 * 7, 7))
    lstm1 = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, return_sequences=True))(input1)
    gru1 = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(256, return_sequences=True))(lstm1)
    flatten = tf.keras.layers.Flatten()(gru1)
    dense = tf.keras.layers.Dense(1)(flatten)
    model = tf.keras.models.Model(inputs=input1, outputs=dense)
    initial_weights = model.get_weights()
    epochs = 10
    lr = 0.00025
    bs = 256
    return model, initial_weights, epochs, lr, bs


def define_model_lstm_hybrid() -> Tuple[
    Tuple[tf.keras.Model, List[np.ndarray], int, float, int],
    Tuple[tf.keras.Model, List[np.ndarray], int, float, int],
]:

    # main model
    inputs = tf.keras.layers.Input((24 * 7 * 6, 13))
    hour_padded_inputs = inputs[:, :, :7]  # first 7 features
    hourly_avg = tf.keras.layers.AveragePooling1D(pool_size=6, strides=6)(
        hour_padded_inputs
    )
    # hourly part of main model has same structure as hourly model and will re-use weights
    hour_lstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True, name="hour1"))(hourly_avg)
    hour_gru = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(128, return_sequences=True, name="hour2"))(hour_lstm)

    # high-frequency part, uses additional std features
    minute_conv1 = tf.keras.layers.Conv1D(
        128,
        kernel_size=6,
        strides=2,
        activation="relu",
        padding="causal",
    )(inputs)
    minute_conv2 = tf.keras.layers.Conv1D(
        128,
        kernel_size=6,
        strides=3,
        activation="relu",
        padding="causal",
    )(minute_conv1)
    min_lstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, return_sequences=True))(minute_conv2)
    min_gru = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(256, return_sequences=True))(min_lstm)
    min_plus_hour = tf.keras.layers.Concatenate()([min_gru, hour_gru])
    drop1 = tf.keras.layers.Dropout(0.2)(min_plus_hour)
    dense = tf.keras.layers.Dense(64, activation="relu")(drop1)
    drop2 = tf.keras.layers.Dropout(0.2)(dense)
    min_out = tf.keras.layers.Dense(1)(tf.keras.layers.Flatten()(drop2))
    minute_model = tf.keras.models.Model(inputs=inputs, outputs=min_out)
    minute_weights = minute_model.get_weights()
    # smaller hourly model, training on longer time period where high-frequency data is not available
    hour_input = tf.keras.layers.Input((7 * 24, 7))
    hour_conv_b = tf.keras.layers.Conv1D(32, kernel_size=6, strides=1)(hour_input)
    hour_lstm_b = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True))(hour_conv_b)
    hour_gru_b = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(128, return_sequences=True))(hour_lstm_b)
    hour_dense_b = tf.keras.layers.Dense(32)(hour_gru_b)
    hour_output = tf.keras.layers.Dense(1)(tf.keras.layers.Flatten()(hour_dense_b))
    hour_model = tf.keras.Model(hour_input, hour_output)
    hour_weights = hour_model.get_weights()
    epochs = 10
    lr = 0.001
    bs = 512
    return (
        (minute_model, minute_weights, epochs, lr, bs),
        (hour_model, hour_weights, epochs, lr, bs),
    )


def define_model_transformer_1_min() -> Tuple[
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
            key_dim=head_size, num_heads=num_heads, dropout=dropout, output_shape=ff_dim,
        )(x, x)
        x = tf.keras.layers.Dropout(dropout)(x)
        res = x + inputs

        # Feed Forward Part
        x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(res)
        x = tf.keras.layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(x)
        x = tf.keras.layers.Dropout(dropout)(x)
        x = tf.keras.layers.Conv1D(filters=ff_dim, kernel_size=1)(x)
        return x + res

    num_transformer_blocks = 4  # how many consecutive transformer layers
    head_size = 128  # channels in the attention head
    inputs = tf.keras.Input((24 * 7 * 6, 13))
    num_heads = 1
    ff_dim = 64
    dropout = 0
    mlp_units = [128]
    mlp_dropout = 0
    x = tf.keras.layers.Conv1D(filters=ff_dim, kernel_size=6, strides=1, activation="relu")(inputs)
    for _ in range(num_transformer_blocks):
        x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout)

    #x = tf.keras.layers.GlobalAveragePooling1D(data_format="channels_first")(x)
    for dim in mlp_units:
        x = tf.keras.layers.Dense(dim, activation="relu")(x)
        x = tf.keras.layers.Dropout(mlp_dropout)(x)
    out_conv = tf.keras.layers.Conv1D(filters=1, kernel_size=1, strides=1)(x)
    output = tf.keras.layers.Dense(1)(tf.keras.layers.Flatten()(out_conv))
    model = tf.keras.Model(inputs, output)
    initial_weights = model.get_weights()
    epochs = 5
    lr = 0.001
    bs = 512
    return model, initial_weights, epochs, lr, bs


def define_model_transformer_hourly() -> Tuple[
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
            key_dim=head_size, num_heads=num_heads, dropout=dropout, output_shape=ff_dim,
        )(x, x)
        x = tf.keras.layers.Dropout(dropout)(x)
        res = x + inputs

        # Feed Forward Part
        x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(res)
        x = tf.keras.layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(x)
        x = tf.keras.layers.Dropout(dropout)(x)
        x = tf.keras.layers.Conv1D(filters=ff_dim, kernel_size=1)(x)
        return x + res

    num_transformer_blocks = 4  # how many consecutive transformer layers
    head_size = 128  # channels in the attention head
    inputs = tf.keras.Input((128, 7))
    timesteps = tf.expand_dims(tf.expand_dims(tf.range(0, 128, delta=1, dtype=float), axis=-1), axis=0)
    ext_inputs = tf.keras.layers.Concatenate()([inputs, timesteps])
    num_heads = 1
    ff_dim = 64
    dropout = 0
    mlp_units = [128]
    mlp_dropout = 0
    x = tf.keras.layers.Conv1D(filters=ff_dim, kernel_size=6, strides=1, activation="relu")(inputs)
    for _ in range(num_transformer_blocks):
        x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout)

    #x = tf.keras.layers.GlobalAveragePooling1D(data_format="channels_first")(x)
    for dim in mlp_units:
        x = tf.keras.layers.Dense(dim, activation="relu")(x)
        x = tf.keras.layers.Dropout(mlp_dropout)(x)
    out_conv = tf.keras.layers.Conv1D(filters=1, kernel_size=1, strides=1)(x)
    output = tf.keras.layers.Dense(1)(tf.keras.layers.Flatten()(out_conv))
    model = tf.keras.Model(inputs, output)
    initial_weights = model.get_weights()
    epochs = 5
    lr = 0.001
    bs = 512

    return model, initial_weights, epochs, lr, bs
