import os
import datetime as dt
import time
from typing import Callable, List, Optional, Tuple, Union

import pandas as pd
from sklearn.metrics import mean_squared_error
import numpy as np
# uncomment to turn off gpu (see https://stackoverflow.com/a/45773574)
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import tensorflow as tf


class DataGen(tf.keras.utils.Sequence):
    """Data generator to dynamically generate batches of training data for keras model.
    Each batch consists of multiple time series sequences.

    See the keras documentation for more details:
    https://keras.io/getting_started/faq/#what-do-sample-batch-and-epoch-mean
    """

    def __init__(
        self,
        x: np.ndarray,
        valid_inds: np.ndarray,
        y: Optional[np.ndarray] = None,
        batch_size: int = 32,
        length: int = 24 * 6 * 7,
        shuffle: bool = True,
    ):
        """Construct the data generator.

        If y is not None, will generate batches of pairs of x and y data, suitable for
        training. If y is None, will generate batches of x data only, suitable for
        prediction.

        x and y data must already be ordered by period and time. The training sample
        generated for an index i in valid_ind will have target y[i] and x variables from
        rows (i - length + 1) to i (inclusive) of x.  If there are multiple periods,
        there should be at least (length - 1) data points before the first valid_ind in
        each period, otherwise the sequence for that valid_ind will include data from
        a previous period.

        Args:
            x: np.array containing the x variables ordered by period and time
            y: None or np.array containing the targets corresponding to the x variables
            batch_size: batch size
            valid_inds: np.array of int containing the indices which are valid
                end-points of training sequences (for example, we may set valid_inds so
                it contains only the data points at the start of each hour).
            length: the number of data points in each sequence of the batch.
            shuffle: whether to shuffle the list of valid_inds before training and after
                each epoch. For training, it is recommended to set this to True, so that
                each batch contains a varied sample of data from different times. For
                prediction, it should be set to False, so that the predicted values are
                in the same order as the input data.
        """

        self.x = x
        self.y = y
        self.length = length
        self.batch_size = batch_size
        self.valid_inds = np.copy(valid_inds)
        self.shuffle = shuffle
        if self.shuffle:
            np.random.shuffle(self.valid_inds)

    def __get_y__(self):
        """Return the array of labels at valid_inds."""
        if self.y is None:
            raise RuntimeError("Generator has no y data.")
        else:
            return self.y[self.valid_inds]

    def __len__(self):
        """Return the number of batches in each epoch."""
        return int(np.ceil(len(self.valid_inds) / self.batch_size))

    def __getitem__(self, idx):
        """Generate a batch. idx is the index of the batch in the training epoch."""
        if (idx < self.__len__() - 1) or (len(self.valid_inds) % self.batch_size == 0):
            num_samples = self.batch_size
        else:
            num_samples = len(self.valid_inds) % self.batch_size
        x = np.empty((num_samples, self.length, self.x.shape[1]))
        end_indexes = self.valid_inds[
            idx * self.batch_size: (idx + 1) * self.batch_size
        ]
        for n, i in enumerate(end_indexes):
            x[n] = self.x[i - self.length + 1 : i + 1, :]
        if self.y is None:
            return x
        else:
            y = self.y[end_indexes]
            return x, y

    def on_epoch_end(self):
        """Code to run at the end of each training epoch."""
        if self.shuffle:
            np.random.shuffle(self.valid_inds)


def define_model_cnn() -> Tuple[tf.keras.Model, np.ndarray, int, float, int]:
    """Define the structure of the neural network.

    Returns:
        keras model
        array of initial weights used to reset the model to its original state
        number of epochs
        learning rate
        batch size
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
        keras model
        array of initial weights used to reset the model to its original state
        number of epochs
        learning rate
        batch size
    """

    inputs = tf.keras.layers.Input((6 * 24 * 7, 13))
    lstm1 = tf.keras.layers.LSTM(50, activation="tanh", return_sequences=True)(inputs)
    lstm2 = tf.keras.layers.LSTM(50, activation="tanh", return_sequences=True)(lstm1)
    lstm3 = tf.keras.layers.LSTM(50, activation="tanh", return_sequences=False)(lstm2)
    dense = tf.keras.layers.Dense(50, activation="tanh")(lstm3)
    output = tf.keras.layers.Dense(1)(dense)
    model = tf.keras.Model(inputs, output)
    initial_weights = model.get_weights()
    epochs = 10
    lr = 0.0005
    bs = 32
    return model, initial_weights, epochs, lr, bs


def prepare_data(
    solar: pd.DataFrame,
    sunspots: Union[pd.DataFrame, float],
    dst: pd.DataFrame = None,
    norm_df=None,
    output_folder: str = None,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Prepare data for training or prediction.

    If dst is None, prepare DataFrame of feature variables only for prediction using
    previously-calculated normalization scaling factors in norm_data_folder. In this
    case norm_factor must not be None. If dst is not None, prepare DataFrame of feature
    variables and labels for model training. Calculate normalization scaling factors and
    save in output_folder. In this case output_folder must not be None.

    Aggregate solar_data into 1-minute intervals and calculate the mean and standard
    deviation. Merge with sunspot data. Normalize the training data and save the scaling
    parameters in a dataframe (these are needed to transform data for prediction).

    This method modifies the input DataFrames solar, sunspots, and dst; if you want to
    keep the original DataFrames, pass copies, e.g. ``prepare_data(solar.copy(),
    sunspots.copy(), dst.copy())``.

    Args:
        solar: DataFrame containing solar wind data
        sunspots: DataFrame containing sunspots data, or float. If DataFrame, will be
            merged with solar data using timestamp. If float, all rows of output data
            will use this number.
        dst: None, or DataFrame containing the disturbance storm time (DST) data, i.e. t
            he labels for training
        norm_df: None, or DataFrame containing the normalization scaling factors to
            apply
        output_folder: Path to the directory where normalisation dataframe will be saved


    Returns:
        DataFrame containing processed data and labels
    """

    # convert timedelta
    solar["timedelta"] = pd.to_timedelta(solar["timedelta"])
    sunspots["timedelta"] = pd.to_timedelta(sunspots["timedelta"])
    sunspots.sort_values(["period", "timedelta"], inplace=True)
    sunspots["month"] = list(range(len(sunspots)))
    sunspots["month"] = sunspots["month"].astype(int)

    # merge data
    solar["days"] = solar["timedelta"].dt.days
    if isinstance(sunspots, pd.DataFrame):
        sunspots["days"] = sunspots["timedelta"].dt.days
        solar = pd.merge(
            solar,
            sunspots[["period", "days", "smoothed_ssn", "month"]],
            "left",
            ["period", "days"],
        )
    else:
        solar["smoothed_ssn"] = sunspots
    solar.drop(columns="days", inplace=True)
    if dst is not None:
        dst["timedelta"] = pd.to_timedelta(dst["timedelta"])
        solar = pd.merge(solar, dst, "left", ["period", "timedelta"])
    solar.sort_values(["period", "timedelta"], inplace=True)
    solar.reset_index(inplace=True, drop=True)

    # remove anomalous data (exclude from training and fill for prediction)
    solar["bad_data"] = False
    solar.loc[solar["temperature"] < 1, "bad_data"] = True
    solar.loc[solar["temperature"] < 1, ["temperature", "speed", "density"]] = np.nan
    for p in ["train_a", "train_b", "train_c"]:
        curr_period = solar["period"] == p
        solar.loc[curr_period, "train_exclude"] = (
            solar.loc[curr_period, "bad_data"].rolling(60 * 24 * 7, center=False).max()
        )

    # fill missing data
    solar["month"] = solar["month"].fillna(method="ffill")
    train_cols = [
        "bt",
        "density",
        "speed",
        "bx_gsm",
        "by_gsm",
        "bz_gsm",
        "smoothed_ssn",
    ]
    train_short = [c for c in train_cols if c != "smoothed_ssn"]
    for p in solar["period"].unique():
        curr_period = solar["period"] == p
        solar.loc[curr_period, "smoothed_ssn"] = (
            solar.loc[curr_period, "smoothed_ssn"]
            .fillna(method="ffill", axis=0)
            .fillna(method="bfill", axis=0)
        )
        roll = (
            solar[train_short]
            .rolling(window=20, min_periods=5)
            .mean()
            .interpolate("linear", axis=0)
        )
        solar.loc[curr_period, train_short] = solar.loc[
            curr_period, train_short
        ].fillna(roll)
        solar.loc[curr_period, train_short] = (
            solar.loc[curr_period, train_short]
            .fillna(method="ffill", axis=0)
            .fillna(method="bfill", axis=0)
        )

    # normalize data using median and inter-quartile range
    if norm_df is None:
        norm_df = solar[train_cols].median().to_frame("median")
        norm_df["lq"] = solar[train_cols].quantile(0.25)
        norm_df["uq"] = solar[train_cols].quantile(0.75)
        norm_df["iqr"] = norm_df["uq"] - norm_df["lq"]
        norm_df.to_csv(os.path.join(output_folder, "norm_df.csv"))
    solar[train_cols] = (solar[train_cols] - norm_df["median"]) / norm_df["iqr"]

    if dst is not None:
        # interpolate target and shift target since we only have data up to t - 1 minute
        solar["target"] = (
            solar["dst"].shift(-1).interpolate(method="linear", limit_direction="both")
        )
        # shift target for training t + 1 hour model
        solar["target_shift"] = solar["target"].shift(-60)
        solar["target_shift"] = solar["target_shift"].fillna(method="ffill")
        assert solar[train_cols + ["target", "target_shift"]].isnull().sum().sum() == 0

    # aggregate features in 10-minute increments
    new_cols = [c + suffix for suffix in ["_mean", "_std"] for c in train_short]
    train_cols = new_cols + ["smoothed_ssn"]
    new_df = pd.DataFrame(index=solar.index, columns=new_cols)
    for p in solar["period"].unique():
        curr_period = solar["period"] == p
        new_df.loc[curr_period] = (
            solar.loc[curr_period, train_short]
            .rolling(window=10, min_periods=1, center=False)
            .agg(["mean", "std"])
            .values
        )
        new_df.loc[curr_period] = (
            new_df.loc[curr_period].fillna(method="ffill").fillna(method="bfill")
        )
    solar = pd.concat([solar, new_df], axis=1)
    solar[train_cols] = solar[train_cols].astype(float)

    # sample at 10-minute frequency
    solar = solar.loc[solar["timedelta"].dt.seconds % 600 == 0].reset_index()

    return solar, train_cols


def train_nn_models(
    solar: pd.DataFrame,
    sunspots: pd.DataFrame,
    dst: pd.DataFrame,
    model_definer: Callable[[], Tuple[tf.keras.Model, np.ndarray, int, float, int]],
    num_models: int = 1,
    output_folder: str = "trained_models",
) -> Optional[List[float]]:
    """Train and save ensemble of models, each trained on a different subset of data.

    Args:
        solar: DataFrame containing solar wind data
        sunspots: DataFrame containing sunspots data
        dst: DataFrame containing the disturbance storm time (DST) data, i.e. the labels
            for training
        model_definer: function returning keras model and initial weights
        num_models: number of models to train.
            Training several models on different subsets of the data and averaging
            results improves model accuracy. If num_models = 1, use all data for
            training. If num_models > 1, each model will be trained on
            int((num_models - 1) / num_models) * num_months randomly-selected months of
            data, where num_months is the total number of months in the training data.
            Because the data contains long-term trends lasting days or weeks, splitting
            the data by month ensures that different models have sufficiently
            different data sets to obtain the benefits of model diversity. Separate
            models are trained for the current and next hour, so the number of models
            output will be num_models * 2.
        output_folder: Path to the directory where models will be saved

    Returns:
        out-of-sample accuracy: If num_models > 1, returns list of length num_models
            containing RMSE values for out-of-sample predictions for each model. These
            provide an indication of model accuracy, but in general will underestimate
            the accuracy of the full ensemble of models. If num_models = 1, returns
            None.
    """

    # prepare data
    solar, train_cols = prepare_data(
        solar, sunspots, dst, output_folder=output_folder, norm_df=None
    )

    # define model and training parameters
    model, initial_weights, epochs, lr, bs = model_definer()
    sequence_length = 6 * 24 * 7

    oos_accuracy = []
    # train on sequences ending at the start of an hour
    valid_bool = solar.index % 6 == 0
    np.random.seed(0)
    solar["month"] = solar["month"].astype(int)
    months = np.sort(solar["month"].unique())
    # remove the first week from each period, because not enough data for prediction
    valid_ind_arr = []
    for p in solar["period"].unique():
        all_p = solar.loc[(solar["period"] == p) & valid_bool].index.values[24 * 7 :]
        valid_ind_arr.append(all_p)
    valid_ind = np.concatenate(valid_ind_arr)
    non_exclude_ind = solar.loc[~solar["train_exclude"].astype(bool)].index.values
    np.random.shuffle(months)
    for model_ind in range(num_models):
        # t model
        tf.keras.backend.clear_session()
        model.compile(
            loss=tf.keras.losses.MeanSquaredError(),
            optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
            metrics=[tf.keras.metrics.RootMeanSquaredError()],
        )
        model.set_weights(initial_weights)
        if num_models > 1:
            # define train and test sets
            leave_out_months = months[
                model_ind
                * (len(months) // num_models) : (model_ind + 1)
                * (len(months) // num_models)
            ]
            leave_out_months_ind = solar.loc[
                valid_bool & solar["month"].isin(leave_out_months)
            ].index.values
            curr_months_ind = solar.loc[
                valid_bool & (~solar["month"].isin(leave_out_months))
            ].index.values
            train_ind = np.intersect1d(
                np.intersect1d(valid_ind, curr_months_ind), non_exclude_ind
            )
            test_ind = np.intersect1d(valid_ind, leave_out_months_ind)
            train_gen = DataGen(
                solar[train_cols].values,
                train_ind,
                solar["target"].values.flatten(),
                bs,
                sequence_length,
            )
            test_gen = DataGen(
                solar[train_cols].values,
                test_ind,
                solar["target"].values.flatten(),
                bs,
                sequence_length,
            )
            model.fit(train_gen, validation_data=test_gen, epochs=epochs, verbose=1)
            oos_accuracy.append(model.evaluate(test_gen, verbose=2)[1])
            print("Out of sample accuracy: ", oos_accuracy)
            print("Out of sample accuracy mean: {}".format(np.mean(oos_accuracy)))
        else:
            # fit on all data
            train_ind = valid_ind
            train_gen = DataGen(
                solar[train_cols].values,
                train_ind,
                solar["target"].values.flatten(),
                bs,
                sequence_length,
            )
            model.fit(train_gen, epochs=epochs, verbose=2)
        model.save_weights(
            os.path.join(output_folder, "model_t_{}.h5".format(model_ind))
        )
        # t + 1 model
        tf.keras.backend.clear_session()
        model.compile(
            loss=tf.keras.losses.MeanSquaredError(),
            optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
            metrics=[tf.keras.metrics.RootMeanSquaredError()],
        )
        model.set_weights(initial_weights)
        data_gen = DataGen(
            solar[train_cols].values,
            train_ind,
            solar["target_shift"].values.flatten(),
            bs,
            sequence_length,
        )
        model.fit(data_gen, epochs=epochs, verbose=2)
        model.save_weights(
            os.path.join(output_folder, "model_t_plus_one_{}.h5".format(model_ind))
        )
        if num_models > 1:
            return oos_accuracy


def load_models(
    input_folder: str, num_models: int
) -> Tuple[List[tf.keras.Model], List[tf.keras.Model], pd.DataFrame]:
    """Define the model structure and load the saved weights of the trained models.

    Args:
        input_folder: path to location where models weights are saved
        num_models: number of models trained for each of t and t + 1 (total number of
            models in folder should be 2 * num_models)

    Returns:
        model_t_arr: List of models for time t
        model_t_plus_one_arr: List of models for time t + 1
        norm_df: DataFrame of scaling factors to normalize the data
    """
    model, _ = define_model_cnn()
    model_t_arr = []
    model_t_plus_one_arr = []
    for i in range(num_models):
        new_model = tf.keras.models.clone_model(model)
        new_model.load_weights(os.path.join(input_folder, "model_t_{}.h5".format(i)))
        model_t_arr.append(new_model)
        new_model = tf.keras.models.clone_model(model)
        new_model.load_weights(
            os.path.join(input_folder, "model_t_plus_one_{}.h5".format(i))
        )
        model_t_plus_one_arr.append(new_model)
    norm_df = pd.read_csv(os.path.join(input_folder, "norm_df.csv"), index_col=0)
    return model_t_arr, model_t_plus_one_arr, norm_df


def predict_one_time(
    solar_wind_7d: pd.DataFrame,
    latest_sunspot_number: float,
    model_t_arr: List[tf.keras.Model],
    model_t_plus_one_arr: List[tf.keras.Model],
    norm_df: pd.DataFrame,
) -> Tuple[float, float]:
    """
    Given 7 days of data at 1-minute frequency up to time t-1, make predictions for
    times t and t+1.

    Args:
        solar_wind_7d: previous 7 days of satellite data up to (t - 1) minutes
        latest_sunspot_number: latest available monthly sunspot number (SSN)
        model_t_arr: List of models for time t
        model_t_plus_one_arr: List of models for time t + 1
        norm_df: Scaling factors to normalize the data

    Returns:
        predictions : predictions for times t and t + 1 hour
    """

    # prepare data
    solar, train_cols = prepare_data(
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
) -> pd.DataFrame:
    """
    Make predictions for multiple times; faster than ``predict_one_time``.

    Input data must be sorted by period and time and have 1-minute frequency, and there
    must be at least 1 week of data before the first prediction time.

    Args:
        solar: DataFrame containing solar wind data
        sunspots: DataFrame containing sunspots data
        prediction_times: DataFrame with a single column `timedelta` for which to make
            predictions. For each value t, return predictions for t and t plus one hour.
        model_t_arr: List of models for time t
        model_t_plus_one_arr: List of models for time t + 1
        norm_df: Scaling factors to normalize the data

    Returns:
        predictions: DataFrame with columns timedelta, period, prediction_t and
            prediction_t_plus_1
    """

    # validate input data
    solar["timedelta"] = pd.to_timedelta(solar["timedelta"])
    diff = solar["timedelta"].diff()
    diff.loc[solar["period"] != solar["period"].shift()] = np.nan
    valid_diff = solar["period"] == solar["period"].shift(1)
    if np.any(diff.loc[valid_diff] != dt.timedelta(minutes=1)):
        raise ValueError(
            "Input data must be sorted by period and time and have 1-minute frequency."
        )

    # add column to solar to indicate which times we must predict
    prediction_times["prediction_time"] = True
    solar = pd.merge(solar, prediction_times, on=["period", "timedelta"], how="left")
    solar["prediction_time"] = solar["prediction_time"].fillna(False)
    solar.sort_values(["period", "timedelta"], inplace=True)
    solar.reset_index(inplace=True, drop=True)

    # prepare data
    solar, train_cols = prepare_data(solar.copy(), sunspots.copy(), norm_df=norm_df)

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
    # these must be 1 minute before the prediction time
    solar["valid_ind"] = solar["prediction_time"].shift(-1).fillna(False)

    # make prediction
    predictions = pd.DataFrame(prediction_times[["timedelta", "period"]].copy())
    valid_ind = solar.loc[solar["valid_ind"]].index.values
    datagen = DataGen(
        solar[train_cols].values,
        valid_ind,
        y=None,
        batch_size=10000,
        length=24 * 6 * 7,
        shuffle=False,
    )
    predictions["prediction_t"] = 0
    for m in model_t_arr:
        predictions["prediction_t"] += np.array(m.predict(datagen)).flatten()
    predictions["prediction_t"] /= len(model_t_arr)
    predictions["prediction_t_plus_1"] = 0
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
