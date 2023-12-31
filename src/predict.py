"""Functions for prediction using pre-trained models."""

import os
import datetime as dt
from typing import List, Tuple

import pandas as pd
import numpy as np
import tensorflow as tf
from keras.losses import mean_squared_error

from preprocessing import (
    DataGen,
    prepare_data_1_min,
    prepare_data_hourly,
    prepare_data_hybrid,
)




def load_models(
    input_folder: str, num_models: int = None,
) -> Tuple[List[tf.keras.Model], List[tf.keras.Model], pd.DataFrame]:
    """Define the model structure and load the saved weights of the trained models.

    Args:
        input_folder: Path to location where model weights are saved
        num_models: Number of models trained for each of ``t`` and ``t + 1`` (total
            number of models in folder should be ``2 * num_models``)

    Returns:
        model_t_arr: List of models for time ``t``
        model_t_plus_one_arr: List of models for time ``t + 1``
        norm_df: DataFrame of scaling factors to normalize the data
    """
    model_t_arr = []
    model_t_plus_one_arr = []
    if num_models is None:
        num_models = len([f for f in os.listdir(input_folder) if f.endswith(".h5")]) // 2
    for i in range(num_models):
        model = tf.keras.models.load_model(
            os.path.join(input_folder, "model_t_{}.h5".format(i))
        )
        model_t_arr.append(model)
        model = tf.keras.models.load_model(
            os.path.join(input_folder, "model_t_plus_one_{}.h5".format(i))
        )
        model_t_plus_one_arr.append(model)
    norm_df = pd.read_csv(os.path.join(input_folder, "norm_df.csv"), index_col=0)
    return model_t_arr, model_t_plus_one_arr, norm_df


def predict_one_time(
    solar_wind_7d: pd.DataFrame,
    latest_sunspot_number: float,
    model_t_arr: List[tf.keras.Model],
    model_t_plus_one_arr: List[tf.keras.Model],
    norm_df: pd.DataFrame,
    frequency: str,
) -> Tuple[float, float]:
    """
    Given 7 days of data at 1-minute frequency up to time ``t-1``, make predictions for
    times ``t`` and ``t+1``.

    Args:
        solar_wind_7d: Previous 7 days of satellite data up to ``(t - 1)`` minutes
        latest_sunspot_number: Latest available monthly sunspot number (SSN)
        model_t_arr: List of models for time ``t``
        model_t_plus_one_arr: List of models for time ``(t + 1)``
        norm_df: Scaling factors to normalize the data
        frequency: frequency of the model, "minute" or "hour"

    Returns:
        prediction_at_t0: Predictions for time ``t``
        prediction_at_t1: Predictions for time ``t + 1``
    """

    # prepare data
    if frequency == "minute":
        solar, train_cols = prepare_data_1_min(
            solar_wind_7d.copy(), latest_sunspot_number, norm_df=norm_df
        )
    else:
        solar, train_cols = prepare_data_hourly(
            solar_wind_7d.copy(), latest_sunspot_number, norm_df=norm_df
        )

    # load model and predict
    pred_data = solar[train_cols].values[np.newaxis, :, :]
    prediction_at_t0 = np.mean(
        [np.array(m.predict(pred_data)).flatten()[0] for m in model_t_arr]
    )
    prediction_at_t1 = np.mean(
        [np.array(m.predict(pred_data)).flatten()[0] for m in model_t_plus_one_arr]
    )

    # restrict to allowed range
    prediction_at_t0 = max(-2000, min(500, prediction_at_t0))
    prediction_at_t1 = max(-2000, min(500, prediction_at_t1))

    return prediction_at_t0, prediction_at_t1


def predict_batch(
    solar: pd.DataFrame,
    sunspots: pd.DataFrame,
    prediction_times: pd.DataFrame,
    model_t_arr: List[tf.keras.Model],
    model_t_plus_one_arr: List[tf.keras.Model],
    norm_df: pd.DataFrame,
    frequency: str,
    comb_model: bool = False,
) -> Tuple[pd.DataFrame, List, List]:
    """
    Make predictions for multiple times; faster than ``predict_one_time``.

    Input data must be sorted by period and time and have 1-minute frequency, and there
    must be at least 1 week of data before the first prediction time.

    Results will be slightly different to ``predict_one_time``, since in that function
    missing data is filled using only data from the previous week.

    Args:
        solar: DataFrame containing solar wind data
        sunspots: DataFrame containing sunspots data
        prediction_times: DataFrame with a single column `timedelta` for which to make
            predictions. For each value ``t``, return predictions for ``t`` and
            ``t`` plus one hour.
        model_t_arr: List of models for time ``t``
        model_t_plus_one_arr: List of models for time ``(t + 1)``
        norm_df: Scaling factors to normalize the data
        frequency: frequency of the model, "minute", "hour" or "hybrid"
        comb_model: if ``True`` assumes a single model that predicts ``t`` and ``t+1``

    Returns:
        predictions: DataFrame with columns ``timedelta``, ``period``, ``prediction_t``
            and ``prediction_t_plus_1``
    """


    # validate input data
    solar["timedelta"] = pd.to_timedelta(solar["timedelta"])
    diff = solar["timedelta"].diff()
    diff.loc[solar["period"] != solar["period"].shift()] = np.nan
    valid_diff = solar["period"] == solar["period"].shift(1)
    if (frequency in ["minute", "hybrid"]) and np.any(
        diff.loc[valid_diff] != dt.timedelta(minutes=1)
    ):
        raise ValueError(
            "Input data must be sorted by period and time and have 1-minute frequency."
        )
    elif (frequency == "hour") and np.any(
        diff.loc[valid_diff] != dt.timedelta(hours=1)
    ):
        raise ValueError(
            "Input data must be sorted by period and time and have 1-hour frequency."
        )

    # add column to solar to indicate which times we must predict
    prediction_times["prediction_time"] = True
    solar = pd.merge(solar, prediction_times, on=["period", "timedelta"], how="left")
    solar["prediction_time"] = solar["prediction_time"].fillna(False)
    solar.sort_values(["period", "timedelta"], inplace=True)
    solar.reset_index(inplace=True, drop=True)

    # prepare data
    if frequency == "minute":
        solar, train_cols = prepare_data_1_min(
            solar.copy(), sunspots.copy(), norm_df=norm_df
        )
    elif frequency == "hour":
        solar, train_cols = prepare_data_hourly(
            solar.copy(), sunspots.copy(), norm_df=norm_df
        )
    elif frequency == "hybrid":
        solar, train_cols = prepare_data_hybrid(
            solar.copy(), None, sunspots.copy(), norm_df=norm_df, freq_for_1_min_data="10_minute"
        )
    else:
        raise ValueError(f"Invalid frequency {frequency}")

    # check there is 1 week of data before each valid time
    min_data_by_period = solar.groupby("period")["timedelta"].min().to_frame("min_time")
    min_data_by_period["min_prediction_time"] = (
        solar.loc[solar["prediction_time"]].groupby("period")["timedelta"].min()
    )
    min_data_by_period["data_before_first_prediction"] = (
        min_data_by_period["min_prediction_time"] - min_data_by_period["min_time"]
    )
    if min_data_by_period["data_before_first_prediction"].min() < dt.timedelta(days=7):
        raise RuntimeError(
            "There must be at least 1 week of data before the first prediction time in each period."
        )

    # valid_ind will be the endpoints of the sequences generated by the data generator;
    # these must be 1 minute/hour before the prediction time
    solar["valid_ind"] = solar["prediction_time"].fillna(False)

    # make prediction
    predictions = pd.DataFrame(prediction_times[["timedelta", "period"]].copy())
    valid_ind = solar.loc[solar["valid_ind"]].index.values
    sequence_length = 24 * 6 * 7 if frequency in ["minute", "hybrid"] else 24 * 7
    datagen = DataGen(
        solar[train_cols].values,
        valid_ind,
        y=None,
        batch_size=100,
        length=sequence_length,
        shuffle=False,
    )

    # Create the array for analyzing five models later.
    t0_predictions_set = []
    t1_predictions_set = []

    predictions["prediction_t"] = 0
    predictions["prediction_t_plus_1"] = 0
    if comb_model:
        for m in model_t_arr:
            predictions[["prediction_t", "prediction_t_plus_1"]] = np.array(m.predict(datagen))
    else:
        for m in model_t_arr:
            curr_model = np.array(m.predict(datagen)).flatten()
            t0_predictions_set.append(curr_model)
            predictions["prediction_t"] += curr_model
        predictions["prediction_t"] /= len(model_t_arr)

        for m in model_t_plus_one_arr:
            curr_model = np.array(m.predict(datagen)).flatten()
            t1_predictions_set.append(curr_model)
            predictions["prediction_t_plus_1"] += curr_model
        predictions["prediction_t_plus_1"] /= len(model_t_plus_one_arr)

    # restrict to allowed range
    predictions["prediction_t"] = np.maximum(
        -2000, np.minimum(500, predictions["prediction_t"])
    )
    predictions["prediction_t_plus_1"] = np.maximum(
        -2000, np.minimum(500, predictions["prediction_t_plus_1"])
    )

    return predictions, t0_predictions_set, t1_predictions_set


def validate_inputs(solar:pd.DataFrame, frequency: str):
    diff = solar["timedelta"].diff()
    diff.loc[solar["period"] != solar["period"].shift()] = np.nan
    valid_diff = solar["period"] == solar["period"].shift(1)
    if (frequency in ["minute", "hybrid"]) and np.any(
            diff.loc[valid_diff] != dt.timedelta(minutes=1)
    ):
        raise ValueError(
            "Input data must be sorted by period and time and have 1-minute frequency."
        )
    elif (frequency == "hour") and np.any(
            diff.loc[valid_diff] != dt.timedelta(hours=1)
    ):
        raise ValueError(
            "Input data must be sorted by period and time and have 1-hour frequency."
        )

def model_evalute(solar:pd.DataFrame, frequency:str):

    solar["timedelta"] = pd.to_timedelta(solar["timedelta"])
    validate_inputs(solar, frequency)






def predict_batch_with_info(
        solar: pd.DataFrame,
        sunspots: pd.DataFrame,
        prediction_times: pd.DataFrame,
        model_t_arr: List[tf.keras.Model],
        model_t_plus_one_arr: List[tf.keras.Model],
        norm_df: pd.DataFrame,
        frequency: str,
        comb_model: bool = False,
) -> pd.DataFrame:
    """
    Make predictions for multiple times; faster than ``predict_one_time``.

    Input data must be sorted by period and time and have 1-minute frequency, and there
    must be at least 1 week of data before the first prediction time.

    Results will be slightly different to ``predict_one_time``, since in that function
    missing data is filled using only data from the previous week.

    Args:
        solar: DataFrame containing solar wind data
        sunspots: DataFrame containing sunspots data
        prediction_times: DataFrame with a single column `timedelta` for which to make
            predictions. For each value ``t``, return predictions for ``t`` and
            ``t`` plus one hour.
        model_t_arr: List of models for time ``t``
        model_t_plus_one_arr: List of models for time ``(t + 1)``
        norm_df: Scaling factors to normalize the data
        frequency: frequency of the model, "minute", "hour" or "hybrid"
        comb_model: if ``True`` assumes a single model that predicts ``t`` and ``t+1``

    Returns:
        predictions: DataFrame with columns ``timedelta``, ``period``, ``prediction_t``
            and ``prediction_t_plus_1``
    """

    # validate input data
    solar["timedelta"] = pd.to_timedelta(solar["timedelta"])
    diff = solar["timedelta"].diff()
    diff.loc[solar["period"] != solar["period"].shift()] = np.nan
    valid_diff = solar["period"] == solar["period"].shift(1)
    if (frequency in ["minute", "hybrid"]) and np.any(
            diff.loc[valid_diff] != dt.timedelta(minutes=1)
    ):
        raise ValueError(
            "Input data must be sorted by period and time and have 1-minute frequency."
        )
    elif (frequency == "hour") and np.any(
            diff.loc[valid_diff] != dt.timedelta(hours=1)
    ):
        raise ValueError(
            "Input data must be sorted by period and time and have 1-hour frequency."
        )

    # add column to solar to indicate which times we must predict
    prediction_times["prediction_time"] = True
    solar = pd.merge(solar, prediction_times, on=["period", "timedelta"], how="left")
    solar["prediction_time"] = solar["prediction_time"].fillna(False)
    solar.sort_values(["period", "timedelta"], inplace=True)
    solar.reset_index(inplace=True, drop=True)

    # prepare data
    if frequency == "minute":
        solar, train_cols = prepare_data_1_min(
            solar.copy(), sunspots.copy(), norm_df=norm_df
        )
    elif frequency == "hour":
        solar, train_cols = prepare_data_hourly(
            solar.copy(), sunspots.copy(), norm_df=norm_df
        )
    elif frequency == "hybrid":
        solar, train_cols = prepare_data_hybrid(
            solar.copy(), None, sunspots.copy(), norm_df=norm_df, freq_for_1_min_data="10_minute"
        )
    else:
        raise ValueError(f"Invalid frequency {frequency}")

    # check there is 1 week of data before each valid time
    min_data_by_period = solar.groupby("period")["timedelta"].min().to_frame("min_time")
    min_data_by_period["min_prediction_time"] = (
        solar.loc[solar["prediction_time"]].groupby("period")["timedelta"].min()
    )
    min_data_by_period["data_before_first_prediction"] = (
            min_data_by_period["min_prediction_time"] - min_data_by_period["min_time"]
    )
    if min_data_by_period["data_before_first_prediction"].min() < dt.timedelta(days=7):
        raise RuntimeError(
            "There must be at least 1 week of data before the first prediction time in each period."
        )

    # valid_ind will be the endpoints of the sequences generated by the data generator;
    # these must be 1 minute/hour before the prediction time
    solar["valid_ind"] = solar["prediction_time"].fillna(False)

    # make prediction
    predictions = pd.DataFrame(prediction_times[["timedelta", "period"]].copy())
    valid_ind = solar.loc[solar["valid_ind"]].index.values
    sequence_length = 24 * 6 * 7 if frequency in ["minute", "hybrid"] else 24 * 7
    datagen = DataGen(
        solar[train_cols].values,
        valid_ind,
        y=None,
        batch_size=100,
        length=sequence_length,
        shuffle=False,
    )
    predictions["prediction_t"] = 0
    predictions["prediction_t_plus_1"] = 0
    if comb_model:
        for m in model_t_arr:
            predictions[["prediction_t", "prediction_t_plus_1"]] = np.array(m.predict(datagen))
    else:
        for m in model_t_arr:
            predictions["prediction_t"] += np.array(m.predict(datagen)).flatten()
        predictions["prediction_t"] /= len(model_t_arr)
        for m in model_t_plus_one_arr:
            predictions["prediction_t_plus_1"] += np.array(m.predict(datagen)).flatten()
        predictions["prediction_t_plus_1"] /= len(model_t_plus_one_arr)

    # restrict to allowed range
    predictions["prediction_t"] = np.maximum(
        -2000, np.minimum(500, predictions["prediction_t"])
    )
    predictions["prediction_t_plus_1"] = np.maximum(
        -2000, np.minimum(500, predictions["prediction_t_plus_1"])
    )

    return predictions

def permutation_importance(solar: pd.DataFrame,
    sunspots: pd.DataFrame,
    dst: pd.DataFrame,
    model_t_arr: List[tf.keras.Model],
    model_t_plus_one_arr: List[tf.keras.Model],
    norm_df: pd.DataFrame,
    train_cols: List) -> pd.DataFrame:



    rmse_permute_df = pd.DataFrame(np.zeros((1, len(train_cols))), columns=train_cols)

    for feature in train_cols:

        test_for_permute = solar.copy(deep=True)
        # Approximate split permutation by simply reversing the data in this feature
        test_for_permute[feature].values[:] = test_for_permute[feature].values[::-1]

        print(f"feature: {feature}")
        # get prediction
        predictions, t0_predictions_set, t1_predictions_set = predict_batch(
            test_for_permute, sunspots, dst, model_t_arr, model_t_plus_one_arr, norm_df, "minute"
        )


        dst_test_1_min = pd.merge(dst, predictions, "left", ["timedelta", "period"])

        loss_t = np.sqrt(
            mean_squared_error(dst_test_1_min["dst"].values, dst_test_1_min["prediction_t"].values)
        )

        rmse_permute_df[feature] = loss_t

        print('%s: %f rmse nano-Tesla' % (feature, rmse_permute_df[feature]))

    return rmse_permute_df





