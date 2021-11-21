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

    input = tf.keras.layers.Input((24 * 7 * 6, 13))
    conv1 = tf.keras.layers.Conv1D(64, kernel_size=6, strides=3, activation="relu")(input)
    conv2 = tf.keras.layers.Conv1D(64, kernel_size=6, strides=3, activation="relu")(conv1)
    drop1 = tf.keras.layers.Dropout(0.2)(conv2)
    lstm1 = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, return_sequences=True))(drop1)
    gru1 = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(128, return_sequences=True))(lstm1)
    dense1 = tf.keras.layers.Dense(32, activation="relu")(gru1)
    dense2 = tf.keras.layers.Dense(32, activation="relu")(dense1)
    drop2 = tf.keras.layers.Dropout(0.2)(dense2)
    flatten = tf.keras.layers.Flatten()(drop2)
    dense3 = tf.keras.layers.Dense(64, activation="relu")(flatten)
    output = tf.keras.layers.Dense(1)(dense3)
    model = tf.keras.models.Model(inputs=input, outputs=output)
    initial_weights = model.get_weights()
    epochs = 7
    lr = 0.00025
    bs = 32
    return model, initial_weights, epochs, lr, bs



def define_model_lstm_hourly() -> Tuple[
    tf.keras.Model, List[np.ndarray], int, float, int
]:

    input = tf.keras.layers.Input((24 * 7, 7))
    lstm1 = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, return_sequences=True))(input)
    gru1 = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(128, return_sequences=True))(lstm1)
    flatten = tf.keras.layers.Flatten()(gru1)
    drop = tf.keras.layers.Dropout(0.2)(flatten)
    dense = tf.keras.layers.Dense(128, activation="relu")(drop)
    output = tf.keras.layers.Dense(1)(dense)
    model = tf.keras.models.Model(inputs=input, outputs=output)
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
    hour_lstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, return_sequences=True, name="hour1"))(hourly_avg)
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
            inputs: tf.Tensor, head_size: int, num_heads: int, ff_dim: int,
            dropout: float, ts_inputs=None
    ) -> tf.Tensor:
        # Normalization and Attention
        if ts_inputs is None:
            ext_inputs = inputs
        else:
            ext_inputs = tf.keras.layers.Concatenate()([inputs, ts_inputs])
        # x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(ext_inputs)
        x = ext_inputs
        x = tf.keras.layers.MultiHeadAttention(
            key_dim=head_size, num_heads=num_heads, dropout=dropout,
            output_shape=ff_dim,
        )(x, x)
        x = tf.keras.layers.Dropout(dropout)(x)
        # res = tf.keras.layers.Concatenate()([x, inputs])
        res = x + tf.keras.layers.Conv1D(ff_dim, kernel_size=1)(inputs)

        # Feed Forward Part
        # x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(res)
        x = tf.keras.layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(x)
        x = tf.keras.layers.Dropout(dropout)(x)
        x = tf.keras.layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(x)
        return x + res

    inputs = tf.keras.Input((24 * 6 * 7, 13))
    conv1 = tf.keras.layers.Conv1D(64, kernel_size=6, strides=3, activation="relu")(
        inputs)
    conv2 = tf.keras.layers.Conv1D(64, kernel_size=6, strides=3, activation="relu")(
        conv1)
    drop1 = tf.keras.layers.Dropout(0.2)(conv2)
    num_ts = 110  # output size of dropout layer

    # positional encoding, from https://www.tensorflow.org/text/tutorials/transformer
    def get_angles(pos, i, d_model):
        angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
        return pos * angle_rates

    angle_rads = get_angles(np.arange(num_ts)[:, np.newaxis],
                            np.arange(32)[np.newaxis, :],
                            num_ts)
    # apply sin to even indices in the array; 2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    # apply cos to odd indices in the array; 2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    pos_encoding = tf.convert_to_tensor(angle_rads, dtype=float)

    timesteps = tf.expand_dims(tf.ones_like(drop1, dtype=float)[:, :, 1],
                               axis=-1) * tf.expand_dims(pos_encoding, axis=0)
    # timesteps_trim = tf.keras.layers.Cropping1D((5, 0))(timesteps)

    num_transformer_blocks = 6  # how many consecutive transformer layers
    head_size = 64  # channels in the attention head
    num_heads = 6
    ff_dim = 64
    dropout = 0.3
    mlp_units = [64]
    mlp_dropout = 0.3
    x = drop1

    # first block has timesteps
    # x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout, timesteps)
    for _ in range(num_transformer_blocks):
        # x = tf.keras.layers.Conv1D(head_size, kernel_size=1)(x) + timesteps
        # x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout)
        x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout, timesteps)
    for dim in mlp_units:
        x = tf.keras.layers.Dense(dim, activation="relu")(x)
        x = tf.keras.layers.Dropout(mlp_dropout)(x)
    out_conv = tf.keras.layers.Conv1D(filters=1, kernel_size=1, strides=1)(x)
    flatten = tf.keras.layers.Flatten()(out_conv)
    output = tf.keras.layers.Dense(1)(flatten)
    #     out_conv = tf.keras.layers.Conv1D(filters=64, kernel_size=1, strides=1)(x)
    #     drop2 = tf.keras.layers.Dropout(0.3)(out_conv)
    #     flatten = tf.keras.layers.Flatten()(drop2)
    #     output = tf.keras.layers.Dense(1)(flatten)
    model = tf.keras.Model(inputs, output)
    initial_weights = model.get_weights()
    epochs = 10
    lr = 0.00005
    bs = 32
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
            inputs: tf.Tensor, head_size: int, num_heads: int, ff_dim: int,
            dropout: float, ts_inputs=None
    ) -> tf.Tensor:
        # Normalization and Attention
        if ts_inputs is None:
            ext_inputs = inputs
        else:
            ext_inputs = tf.keras.layers.Concatenate()([inputs, ts_inputs])
        # x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(ext_inputs)
        x = ext_inputs
        x = tf.keras.layers.MultiHeadAttention(
            key_dim=head_size, num_heads=num_heads, dropout=dropout,
            output_shape=ff_dim,
        )(x, x)
        x = tf.keras.layers.Dropout(dropout)(x)
        # res = tf.keras.layers.Concatenate()([x, inputs])
        res = x + tf.keras.layers.Conv1D(ff_dim, kernel_size=1)(inputs)

        # Feed Forward Part
        # x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(res)
        x = tf.keras.layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(x)
        x = tf.keras.layers.Dropout(dropout)(x)
        x = tf.keras.layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(x)
        return x + res

    inputs = tf.keras.Input((24 * 7, 7))
    num_ts = 24 * 7

    # positional encoding, from https://www.tensorflow.org/text/tutorials/transformer
    def get_angles(pos, i, d_model):
        angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
        return pos * angle_rates

    angle_rads = get_angles(np.arange(num_ts)[:, np.newaxis],
                            np.arange(32)[np.newaxis, :],
                            num_ts)
    # apply sin to even indices in the array; 2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    # apply cos to odd indices in the array; 2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    pos_encoding = tf.convert_to_tensor(angle_rads, dtype=float)

    timesteps = tf.expand_dims(tf.ones_like(inputs, dtype=float)[:, :, 1],
                               axis=-1) * tf.expand_dims(pos_encoding, axis=0)
    num_transformer_blocks = 6  # how many consecutive transformer layers
    head_size = 64  # channels in the attention head
    num_heads = 6
    ff_dim = 64
    dropout = 0.3
    mlp_units = [64]
    mlp_dropout = 0.3
    x = inputs

    # first block has timesteps
    for _ in range(num_transformer_blocks):
        x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout, timesteps)
    for dim in mlp_units:
        x = tf.keras.layers.Dense(dim, activation="relu")(x)
        x = tf.keras.layers.Dropout(mlp_dropout)(x)
    out_conv = tf.keras.layers.Conv1D(filters=1, kernel_size=1, strides=1)(x)
    flatten = tf.keras.layers.Flatten()(out_conv)
    output = tf.keras.layers.Dense(1)(flatten)
    model = tf.keras.Model(inputs, output)
    initial_weights = model.get_weights()
    epochs = 10
    lr = 0.00005
    bs = 32
    return model, initial_weights, epochs, lr, bs
