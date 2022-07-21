"""Function to benchmark the model by training on private and predicting on public.
Functionality is similar to the ``benchmark.ipynb`` notebook, but in script form, so it
can be incorporated into a larger workflow. (For example, training several models and
generating a table to compare performance.)"""

import os
import datetime as dt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from model_definitions import define_model_cnn
from predict import load_models, predict_batch
from train import train_nn_models
import time
from typing import Callable, Tuple
import tensorflow as tf


def benchmark(
    model_definition: Callable[[], Tuple[tf.keras.Model, np.ndarray, int, float, int]],
    model_name: str,
) -> Tuple[float, float, float, float]:
    """Train model on public data, predict on private and measure performance.

    Prediction is done with the ``predict_batch`` method, rather than repeated calls to
    ``predict_one_time``, to reduce prediction time. Results will be slightly different;
    see the documentation of ``predict_batch`` for details.

    Args:
        model_definition: function returning a keras model and training parameters, as
            found in the ``model_definitions`` module.
        model_name: name of the sub-folder in which to save the model. The full path
            will be ``os.path.join('trained_models', model_name, 'benchmark')``.

    Returns:
        loss_t: RMSE on private dataset for time ``t``
        loss_t_plus_one: RMSE on private dataset for time ``t + 1``
        training_time: time in seconds to train the model
        prediction_time: time in secons to make the predicition
    """
    # load data
    data_folder = "test_data"
    solar_train = pd.read_csv(os.path.join(data_folder, "public", "solar_wind.csv"))
    dst_train = pd.read_csv(os.path.join(data_folder, "public", "dst_labels.csv"))
    sunspots_train = pd.read_csv(os.path.join(data_folder, "public", "sunspots.csv"))
    solar_test = pd.read_csv(os.path.join(data_folder, "private", "solar_wind.csv"))
    dst_test = pd.read_csv(os.path.join(data_folder, "private", "dst_labels.csv"))
    sunspots_test = pd.read_csv(os.path.join(data_folder, "private", "sunspots.csv"))

    # train and save models
    output_folder = os.path.join("trained_models_test", model_name, "benchmark")
    os.makedirs(output_folder, exist_ok=True)
    t = time.time()
    train_nn_models(
        solar_train, sunspots_train, dst_train, define_model_cnn, 1, output_folder
    )
    training_time = time.time() - t
    # measure performance on train and test
    t = time.time()
    model_t_arr, model_t_plus_1_arr, norm_df = load_models(output_folder, 1)
    dst_test["timedelta"] = pd.to_timedelta(dst_test["timedelta"])
    # exclude times in the first week of dst_test
    dst_test = dst_test.loc[dst_test["timedelta"] >= dt.timedelta(days=7)]
    predictions = predict_batch(
        solar_test, sunspots_test, dst_test, model_t_arr, model_t_plus_1_arr, norm_df
    )
    dst_test = pd.merge(dst_test, predictions, "left", ["timedelta", "period"])
    dst_test["dst_t_plus_1"] = dst_test.groupby("period")["dst"].shift(-1)
    loss_t = np.sqrt(
        mean_squared_error(dst_test["dst"].values, dst_test["prediction_t"].values)
    )
    valid_ind = dst_test["dst_t_plus_1"].notnull()
    loss_t_plus_1 = np.sqrt(
        mean_squared_error(
            dst_test.loc[valid_ind, "dst_t_plus_1"].values,
            dst_test.loc[valid_ind, "prediction_t_plus_1"].values,
        )
    )
    prediction_time = time.time() - t
    return loss_t, loss_t_plus_1, training_time, prediction_time


if __name__ == "__main__":
    loss_t, loss_t_plus_1, training_time, prediction_time = benchmark(
        define_model_cnn, "cnn"
    )
    print(f"RMSE for time t: {loss_t:0.2f}")
    print(f"RMSE for time t+1: {loss_t_plus_1:0.2f}")
    print(f"Training time: {training_time:0.1f}s")
    print(f"Prediction time: {prediction_time:0.1f}s")
