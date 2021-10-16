"""Definitions of keras models."""

from typing import Tuple

import numpy as np
import tensorflow as tf


def define_model_cnn() -> Tuple[tf.keras.Model, np.ndarray, int, float, int]:
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
    lr = 0.0005
    bs = 32
    return model, initial_weights, epochs, lr, bs
