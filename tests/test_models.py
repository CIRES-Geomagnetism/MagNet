import datetime as dt
import pytest
import pandas as pd
import numpy as np
import os
from preprocessing import prepare_data
from predict import predict_batch, predict_one_time
import tensorflow as tf


@pytest.fixture
def test_data(tmpdir):
    """Load small dataset and calculate normalization factors."""
    data_folder = os.path.join("..", "test_data")
    solar_train = pd.read_csv(os.path.join(data_folder, "public", "solar_wind.csv"))
    dst_train = pd.read_csv(os.path.join(data_folder, "public", "dst_labels.csv"))
    sunspots_train = pd.read_csv(os.path.join(data_folder, "public", "sunspots.csv"))
    _, _ = prepare_data(solar_train, sunspots_train, dst_train, output_folder=tmpdir)
    norm_df = pd.read_csv(os.path.join(tmpdir, "norm_df.csv"), index_col=0)
    return solar_train, sunspots_train, dst_train, norm_df


@pytest.fixture
def simple_models(test_data):
    """Create a simple model with random weights."""
    inputs = tf.keras.layers.Input((6 * 24 * 7, 13))
    dense = tf.keras.layers.Dense(1, activation="relu")(inputs)
    output = tf.keras.layers.Dense(1, activation="relu")(
        tf.keras.layers.Flatten()(dense)
    )
    model = tf.keras.Model(inputs, output)
    model.set_weights([np.random.random(w.shape) * 0.1 for w in model.weights])
    model_plus_one = tf.keras.Model(inputs, output)
    model_plus_one.set_weights(
        [np.random.random(w.shape) * 0.1 for w in model_plus_one.weights]
    )
    return model, model_plus_one


def test_no_nulls_in_prepared_data(test_data, tmpdir):
    """Test that output of prepare_data contains no nulls in training or target
    columns."""
    solar, sunspots, dst, norm_df = test_data
    df, train_cols = prepare_data(solar, sunspots, dst, output_folder=tmpdir)
    assert df[train_cols + ["target", "target_shift"]].notnull().all().all()


def test_predict_one_time_vs_predict_batch(test_data, simple_models):
    """Test that predict_one_time and predict_batch give same result if
    there is no missing data."""
    solar, sunspots, dst, norm_df = test_data
    solar = solar.fillna(method="ffill").fillna(method="bfill")
    solar.loc[solar["temperature"] <= 1, "temperature"] = 10
    model, model_plus_one = simple_models
    prediction_times = dst.loc[dst["timedelta"] >= dt.timedelta(days=7)].copy()
    batch_predictions = predict_batch(
        solar, sunspots, prediction_times, [model], [model_plus_one], norm_df
    )
    # only test 5 rows, otherwise too slow
    batch_predictions = batch_predictions.iloc[:5]
    one_time_predictions = pd.DataFrame(
        index=batch_predictions.index, columns=batch_predictions.columns
    )
    one_time_predictions[["timedelta", "period"]] = batch_predictions[
        ["timedelta", "period"]
    ]
    one_time_predictions[
        ["prediction_t", "prediction_t_plus_1"]
    ] = one_time_predictions[["prediction_t", "prediction_t_plus_1"]].astype(float)
    for i in one_time_predictions.index:
        row = one_time_predictions.loc[i]
        t = row["timedelta"]
        period = row["period"]
        time_filter = (solar["timedelta"] < t) & (
            solar["timedelta"] >= (t - dt.timedelta(days=7))
        )
        period_filter = solar["period"] == period
        solar_wind_7d = solar.loc[time_filter & period_filter]
        latest_sunspot_number = sunspots.loc[
            (sunspots["timedelta"] <= t) & (sunspots["period"] == period),
            "smoothed_ssn",
        ].values[-1]
        pred_t, pred_t_plus_1 = predict_one_time(
            solar_wind_7d, latest_sunspot_number, [model], [model_plus_one], norm_df
        )
        one_time_predictions.loc[i, ["prediction_t", "prediction_t_plus_1"]] = [
            pred_t,
            pred_t_plus_1,
        ]
    np.testing.assert_allclose(
        batch_predictions[["prediction_t", "prediction_t_plus_1"]].values,
        one_time_predictions[["prediction_t", "prediction_t_plus_1"]].values,
        atol=1e-3,
        rtol=1e-3,
    )
