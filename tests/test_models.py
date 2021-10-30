import datetime as dt
import pytest
import pandas as pd
import numpy as np
import os
from preprocessing import prepare_data_1_min, prepare_data_hourly, combine_old_and_new_data
from predict import predict_batch, predict_one_time
from train import train_nn_models
import tensorflow as tf


@pytest.fixture
def raw_data():
    data_folder = os.path.join("..", "test_data")
    solar_train = pd.read_csv(os.path.join(data_folder, "public", "solar_wind.csv"))
    dst_train = pd.read_csv(os.path.join(data_folder, "public", "dst_labels.csv"))
    sunspots_train = pd.read_csv(os.path.join(data_folder, "public", "sunspots.csv"))
    return solar_train, sunspots_train, dst_train


@pytest.fixture
def prepared_data_1_min(raw_data, tmpdir):
    """Load small dataset and calculate normalization factors."""
    solar_train, sunspots_train, dst_train = raw_data
    _, _ = prepare_data_1_min(solar_train, sunspots_train, dst_train, output_folder=tmpdir)
    norm_df = pd.read_csv(os.path.join(tmpdir, "norm_df.csv"), index_col=0)
    return solar_train, sunspots_train, dst_train, norm_df


@pytest.fixture
def prepared_data_hourly(raw_data, tmpdir):
    """Load small dataset and calculate normalization factors."""
    solar_train, sunspots_train, dst_train = raw_data
    solar_train["timedelta"] = pd.to_timedelta(solar_train["timedelta"])
    solar_train = solar_train.loc[solar_train["timedelta"].dt.seconds % 3600 == 0]
    _, _ = prepare_data_hourly(solar_train, sunspots_train, dst_train, output_folder=tmpdir)
    norm_df = pd.read_csv(os.path.join(tmpdir, "norm_df.csv"), index_col=0)
    return solar_train, sunspots_train, dst_train, norm_df


def simple_model_definer_1_min():
    inputs = tf.keras.layers.Input((6 * 24 * 7, 13))
    dense = tf.keras.layers.Dense(1, activation="relu")(inputs)
    output = tf.keras.layers.Dense(1, activation="relu")(
        tf.keras.layers.Flatten()(dense)
    )
    model = tf.keras.Model(inputs, output)
    initial_weights = model.get_weights()
    epochs = 1
    lr = 0.00025
    bs = 32
    return model, initial_weights, epochs, lr, bs


def simple_model_definer_hourly():
    inputs = tf.keras.layers.Input((24 * 7, 6))
    dense = tf.keras.layers.Dense(1, activation="relu")(inputs)
    output = tf.keras.layers.Dense(1, activation="relu")(
        tf.keras.layers.Flatten()(dense)
    )
    model = tf.keras.Model(inputs, output)
    initial_weights = model.get_weights()
    epochs = 1
    lr = 0.00025
    bs = 32
    return model, initial_weights, epochs, lr, bs


@pytest.fixture
def simple_models_1_min():
    """Create a simple model with random weights."""
    model = simple_model_definer_1_min()[0]
    model.set_weights([np.random.random(w.shape) * 0.1 for w in model.weights])
    model_plus_one = tf.keras.models.clone_model(model)
    model_plus_one.set_weights(
        [np.random.random(w.shape) * 0.1 for w in model_plus_one.weights]
    )
    return model, model_plus_one


@pytest.fixture
def simple_models_hourly():
    """Create a simple model with random weights."""
    model = simple_model_definer_hourly()[0]
    model.set_weights([np.random.random(w.shape) * 0.1 for w in model.weights])
    model_plus_one = tf.keras.models.clone_model(model)
    model_plus_one.set_weights(
        [np.random.random(w.shape) * 0.1 for w in model_plus_one.weights]
    )
    return model, model_plus_one


def test_load(simple_models_1_min, prepared_data_1_min, tmpdir):
    m1, m2 = simple_models_1_min
    m1.save(os.path.join(tmpdir, "model_t_0.h5"))
    m2.save(os.path.join(tmpdir, "model_t_plus_one_0.h5"))
    m1_new = tf.keras.models.load_model(os.path.join(tmpdir, "model_t_0.h5"))
    m2_new = tf.keras.models.load_model(os.path.join(tmpdir, "model_t_plus_one_0.h5"))
    # test that predicting with reloaded model give same result
    solar, sunspots, dst, norm_df = prepared_data_1_min
    prediction_times = dst.loc[dst["timedelta"] >= dt.timedelta(days=7)].copy()
    pred1 = predict_batch(solar, sunspots, prediction_times, [m1], [m2], norm_df, "minute")
    pred2 = predict_batch(
        solar, sunspots, prediction_times, [m1_new], [m2_new], norm_df, "minute"
    )
    np.testing.assert_allclose(
        pred1[["prediction_t", "prediction_t_plus_1"]].values,
        pred2[["prediction_t", "prediction_t_plus_1"]].values,
    )


@pytest.mark.parametrize('frequency, prepared_data, model_definer',
                         [('minute', 'prepared_data_1_min', simple_model_definer_1_min),
                          ('hour', 'prepared_data_hourly', simple_model_definer_hourly)])
def test_train(raw_data, prepared_data, model_definer, frequency, request, tmpdir):
    solar, sunspots, dst = raw_data
    train_nn_models(solar, sunspots, dst, model_definer, 1, tmpdir, frequency)
    # test models can be loaded and used for prediction
    m1 = tf.keras.models.load_model(os.path.join(tmpdir, "model_t_0.h5"))
    m2 = tf.keras.models.load_model(os.path.join(tmpdir, "model_t_plus_one_0.h5"))
    # test that predicting with reloaded model gives same result
    solar, sunspots, dst, norm_df = request.getfixturevalue(prepared_data)
    prediction_times = dst.loc[dst["timedelta"] >= dt.timedelta(days=7)].copy()
    pred = predict_batch(solar, sunspots, prediction_times, [m1], [m2], norm_df, frequency)
    assert (
        np.isnan(pred[["prediction_t", "prediction_t_plus_1"]].values).sum().sum() == 0
    )


@pytest.mark.parametrize('prepared_data', ['prepared_data_1_min', 'prepared_data_hourly'])
def test_no_nulls_in_prepared_data(prepared_data, request, tmpdir):
    """Test that output of prepare_data contains no nulls in training or target
    columns."""
    solar, sunspots, dst, norm_df = request.getfixturevalue(prepared_data)
    df, train_cols = prepare_data_1_min(solar, sunspots, dst, output_folder=tmpdir)
    assert df[train_cols + ["target", "target_shift"]].notnull().all().all()


@pytest.mark.parametrize('frequency, prepared_data, simple_models',
                         [('minute', 'prepared_data_1_min', 'simple_models_1_min'),
                          ('hour', 'prepared_data_hourly', 'simple_models_hourly')])
def test_predict_one_time_vs_predict_batch(prepared_data, simple_models, frequency, request):
    """Test that predict_one_time and predict_batch give same result if
    there is no missing data."""
    solar, sunspots, dst, norm_df = request.getfixturevalue(prepared_data)
    solar = solar.fillna(method="ffill").fillna(method="bfill")
    solar.loc[solar["temperature"] <= 1, "temperature"] = 10
    model, model_plus_one = request.getfixturevalue(simple_models)
    prediction_times = dst.loc[dst["timedelta"] >= dt.timedelta(days=7)].copy()
    batch_predictions = predict_batch(
        solar, sunspots, prediction_times, [model], [model_plus_one], norm_df, frequency
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
            solar_wind_7d, latest_sunspot_number, [model], [model_plus_one], norm_df, frequency
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


def test_combine_data():
    ind = pd.to_timedelta(np.arange(24 * 60), unit="minute")
    data = np.arange(24 * 60)
    new_data = pd.DataFrame({'timedelta': ind, 'data': data})
    new_data["period"] = "a"
    old_data = pd.DataFrame()
    comb_data = combine_old_and_new_data(old_data, new_data)
    assert comb_data.loc[dt.timedelta(seconds=3600), "data"] == np.mean(data[:60])
    assert comb_data.loc[dt.timedelta(seconds=3600 * 24), "data"] == np.mean(data[-60:])
