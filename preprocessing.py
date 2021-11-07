"""Functions to pre-process raw data before training or prediciton."""

import os
from typing import List, Optional, Tuple, Union

import pandas as pd
import numpy as np
import tensorflow as tf


def prepare_data_1_min(
    solar: pd.DataFrame,
    sunspots: Union[pd.DataFrame, float],
    dst: pd.DataFrame = None,
    norm_df=None,
    output_folder: str = None,
    coord_system: str = "gsm",
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Prepare data for training or prediction.

    If ``dst`` is ``None``, prepare dataframe of feature variables only for prediction
    using previously-calculated normalization scaling factors in ``norm_df``.
    If ``dst`` is not ``None``, prepare dataframe of feature variables and labels for
    model training. Calculate normalization scaling factors and
    save in ``output_folder``. In this case ``output_folder`` must not be ``None``.

    Aggregate solar_data into 10-minute intervals and calculate the mean and standard
    deviation. Merge with sunspot data. Normalize the training data and save the scaling
    parameters in a dataframe (these are needed to transform data for prediction).

    This method modifies the input dataframes ``solar``, ``sunspots``, and ``dst``; if
    you want to keep the original dataframes, pass copies, e.g.
    ``prepare_data_1_min(solar.copy(), sunspots.copy(), dst.copy())``.

    Args:
        solar: DataFrame containing solar wind data. This function uses the GSM
            co-ordinates.
        sunspots: DataFrame containing sunspots data, or float. If dataframe, will be
            merged with solar data using timestamp. If float, all rows of output data
            will use this number.
        dst: ``None``, or DataFrame containing the disturbance storm time (DST) data,
        i.e. the labels for training
        norm_df: ``None``, or DataFrame containing the normalization scaling factors to
            apply
        output_folder: Path to the directory where normalisation dataframe will be saved
        coord_system: either "gsm" or "gse"


    Returns:
        solar: DataFrame containing processed data and labels
        train_cols: list of training columns
    """

    # convert timedelta
    solar["timedelta"] = pd.to_timedelta(solar["timedelta"])

    # merge data
    solar["days"] = solar["timedelta"].dt.days
    if isinstance(sunspots, pd.DataFrame):
        sunspots["timedelta"] = pd.to_timedelta(sunspots["timedelta"])
        sunspots.sort_values(["period", "timedelta"], inplace=True)
        sunspots["month"] = list(range(len(sunspots)))
        sunspots["month"] = sunspots["month"].astype(int)
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
    for p in solar["period"].unique():
        curr_period = solar["period"] == p
        solar.loc[curr_period, "train_exclude"] = (
            solar.loc[curr_period, "bad_data"].rolling(60 * 24 * 7, center=False).max()
        )

    # fill missing data
    if "month" in solar.columns:
        solar["month"] = solar["month"].fillna(method="ffill")
    if coord_system == "gsm":
        train_cols = [
            "bt",
            "density",
            "speed",
            "bx_gsm",
            "by_gsm",
            "bz_gsm",
            "smoothed_ssn",
        ]
    elif coord_system == "gse":
        train_cols = [
            "bt",
            "density",
            "speed",
            "bx_gse",
            "by_gse",
            "bz_gse",
            "smoothed_ssn",
        ]
    else:
        raise ValueError(f"Invalid coord system {coord_system}")
    train_short = [c for c in train_cols if c != "smoothed_ssn"]
    for p in solar["period"].unique():
        curr_period = solar["period"] == p
        solar.loc[curr_period, "smoothed_ssn"] = (
            solar.loc[curr_period, "smoothed_ssn"]
            .fillna(method="ffill", axis=0)
            .fillna(method="bfill", axis=0)
        )
        # fill short gaps with interpolation
        roll = (
            solar[train_short]
            .rolling(window=20, min_periods=5)
            .mean()
            .interpolate("linear", axis=0, limit=60)
        )
        solar.loc[curr_period, train_short] = solar.loc[
            curr_period, train_short
        ].fillna(roll)
        solar.loc[curr_period, train_short] = solar.loc[
            curr_period, train_short
        ].fillna(solar.loc[curr_period, train_short].mean(), axis=0)

    # normalize data using median and inter-quartile range
    if norm_df is None:
        norm_df = solar[train_cols].median().to_frame("median")
        norm_df["lq"] = solar[train_cols].quantile(0.25)
        norm_df["uq"] = solar[train_cols].quantile(0.75)
        norm_df["iqr"] = norm_df["uq"] - norm_df["lq"]
    if output_folder is not None:
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


def prepare_data_hourly(
    solar: pd.DataFrame,
    sunspots: Union[pd.DataFrame, float],
    dst: pd.DataFrame = None,
    norm_df=None,
    output_folder: str = None,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Prepare data for training or prediction.

    If ``dst`` is ``None``, prepare dataframe of feature variables only for prediction
    using previously-calculated normalization scaling factors in ``norm_df``.
    If ``dst`` is not ``None``, prepare dataframe of feature variables and labels for
    model training. Calculate normalization scaling factors and
    save in ``output_folder``. In this case ``output_folder`` must not be ``None``.

    Normalize the training data and save the scaling parameters in a dataframe (these
    are needed to transform data for prediction).

    This method modifies the input dataframes ``solar``, ``sunspots``, and ``dst``; if
    you want to keep the original dataframes, pass copies, e.g.
    ``prepare_data_hourly(solar.copy(), sunspots.copy(), dst.copy())``.

    Args:
        solar: DataFrame containing solar wind data. This function uses the GSE
            co-ordinates.
        sunspots: DataFrame containing sunspots data, or float. If dataframe, will be
            merged with solar data using timestamp. If float, all rows of output data
            will use this number.
        dst: ``None``, or DataFrame containing the disturbance storm time (DST) data,
        i.e. the labels for training
        norm_df: ``None``, or DataFrame containing the normalization scaling factors to
            apply
        output_folder: Path to the directory where normalisation dataframe will be saved


    Returns:
        solar: DataFrame containing processed data and labels
        train_cols: list of training columns
    """

    # convert timedelta
    solar["timedelta"] = pd.to_timedelta(solar["timedelta"])

    # calculate bt
    solar["bt"] = np.sqrt(
        solar["bx_gse"] ** 2 + solar["by_gse"] ** 2 + solar["bz_gse"] ** 2
    )

    # merge data
    solar["days"] = solar["timedelta"].dt.days
    if isinstance(sunspots, pd.DataFrame):
        sunspots["timedelta"] = pd.to_timedelta(sunspots["timedelta"])
        sunspots.sort_values(["period", "timedelta"], inplace=True)
        sunspots["month"] = list(range(len(sunspots)))
        sunspots["month"] = sunspots["month"].astype(int)
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
    for p in solar["period"].unique():
        curr_period = solar["period"] == p
        solar.loc[curr_period, "train_exclude"] = (
            solar.loc[curr_period, "bad_data"].rolling(24 * 7, center=False).max()
        )

    # fill missing data
    if "month" in solar.columns:
        solar["month"] = solar["month"].fillna(method="ffill")
    train_cols = [
        "bt",
        "density",
        "speed",
        "bx_gse",
        "by_gse",
        "bz_gse",
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
        # fill short gaps with interpolation
        roll = (
            solar[train_short]
            .rolling(window=20, min_periods=1)
            .mean()
            .interpolate("linear", axis=0, limit=1)
        )
        solar.loc[curr_period, train_short] = solar.loc[
            curr_period, train_short
        ].fillna(roll)
        solar.loc[curr_period, train_short] = solar.loc[
            curr_period, train_short
        ].fillna(solar.loc[curr_period, train_short].mean(), axis=0)

    # normalize data using median and inter-quartile range
    if norm_df is None:
        norm_df = solar[train_cols].median().to_frame("median")
        norm_df["lq"] = solar[train_cols].quantile(0.25)
        norm_df["uq"] = solar[train_cols].quantile(0.75)
        norm_df["iqr"] = norm_df["uq"] - norm_df["lq"]
    if output_folder is not None:
        norm_df.to_csv(os.path.join(output_folder, "norm_df.csv"))
    solar[train_cols] = (solar[train_cols] - norm_df["median"]) / norm_df["iqr"]

    if dst is not None:
        # interpolate target and shift target since we only have data up to t - 1 minute
        solar["target"] = (
            solar["dst"].shift(-1).interpolate(method="linear", limit_direction="both")
        )
        # shift target for training t + 1 hour model
        solar["target_shift"] = solar["target"].shift(-1)
        solar["target_shift"] = solar["target_shift"].fillna(method="ffill")

    train_cols = [
        "density",
        "speed",
        "bx_gse",
        "by_gse",
        "bz_gse",
        "bt",
        "smoothed_ssn",
    ]
    solar[train_cols] = solar[train_cols].astype(float)

    return solar, train_cols


def prepare_data_hybrid(
    solar_1_min: pd.DataFrame,
    solar_hourly: Optional[pd.DataFrame],
    sunspots: Union[pd.DataFrame, float],
    dst: pd.DataFrame = None,
    norm_df=None,
    output_folder: str = None,
    output_folder_hourly: str = None,
) -> Union[
    Tuple[pd.DataFrame, pd.DataFrame, List[str], List[str]],
    Tuple[pd.DataFrame, List[str]],
]:
    """
    Prepare data for training or prediction. Returns both hourly and 1-minute data.
    In the hourly data, the features ``bx_gse_mean``, ``by_gse_mean``, ``bz_gse_mean``
    ``bt``, ``speed``, and ``density`` are normalised using the same normalisation
    factors as the 1-minute data. This means the 1-minute data can be used in a
    module trained on the hourly data.

    If ``dst`` is ``None``, prepare dataframe of feature variables only for prediction
    using previously-calculated normalization scaling factors in ``norm_df``.
    If ``dst`` is not ``None``, prepare dataframe of feature variables and labels for
    model training. Calculate normalization scaling factors and
    save in ``output_folder``. In this case ``output_folder`` must not be ``None``.

    Normalize the training data and save the scaling parameters in a dataframe (these
    are needed to transform data for prediction).

    This method modifies the input dataframes ``solar_hourly``, ``solar``, ``sunspots``,
    and ``dst``; if you want to keep the original dataframes, pass copies, e.g.
    ``prepare_data_hybrid(solar.copy(), sunspots.copy(), dst.copy())``.

    Args:
        solar_1_min: DataFrame containing solar wind data at 1-minute frequency. This
            function uses the GSE co-ordinates.
        solar_hourly: DataFrame containing solar wind data at hourly frequency. This
            can contain data from the same time period as solar_1_min. If None,
            only return the 1-minute data, but order the columns correctly for using
            the hybrid model.
        sunspots: DataFrame containing sunspots data, or float. If dataframe, will be
            merged with solar data using timestamp. If float, all rows of output data
            will use this number.
        dst: ``None``, or DataFrame containing the disturbance storm time (DST) data,
        i.e. the labels for training
        norm_df: ``None``, or DataFrame containing the normalization scaling factors to
            apply
        output_folder: Path to the directory where normalisation dataframe for 1-minute
            data will be saved
        output_folder_hourly: Path to the directory where normalisation dataframe for 1-minute
            data will be saved

    Returns:
        solar_1_min, solar_hourly, train_cols_1_min, train_cols_hourly: if `
            `solar_hourly`` is not None
        solar_1_min, train_cols_1_min: if ``solar_hourly`` is None
    """

    # prepare 1-minute data
    solar_1_min, train_cols_1_min = prepare_data_1_min(
        solar_1_min,
        sunspots,
        dst,
        output_folder=output_folder,
        norm_df=norm_df,
        coord_system="gse",
    )

    common_columns = [
        "bx_gse_mean",
        "by_gse_mean",
        "bz_gse_mean",
        "bt_mean",
        "density_mean",
        "speed_mean",
        "smoothed_ssn",
    ]
    common_columns_no_suffix = [c.replace("_mean", "") for c in common_columns]
    if solar_hourly is not None:
        # prepare hourly data, using same normalisation factors
        norm_df = pd.read_csv(os.path.join(output_folder, "norm_df.csv"), index_col=0)
        norm_df = norm_df.loc[common_columns_no_suffix]
        solar_hourly, train_cols_hourly = prepare_data_hourly(
            solar_hourly,
            sunspots,
            dst,
            output_folder=output_folder_hourly,
            norm_df=norm_df,
        )

    # sort columns so that common columns occur in same order in 1-minute and
    # hourly data
    minute_ordered = common_columns + [
        c for c in train_cols_1_min if c not in common_columns
    ]
    hour_ordered = common_columns_no_suffix

    if solar_hourly is None:
        return solar_1_min, minute_ordered
    else:
        return solar_1_min, solar_hourly, minute_ordered, hour_ordered


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

        If ``y`` is not ``None``, will generate batches of pairs of x and y data,
        suitable for training. If ``y`` is ``None``, will generate batches of ``x``
        data only, suitable for prediction.

        ``x`` and ``y`` data must already be ordered by period and time. The training
        sample generated for an index ``i`` in ``valid_ind`` will have target ``y[i]``
        and ``x`` variables from rows ``(i - length + 1)`` to ``i`` (inclusive) of
        ``x``.  If there are multiple periods, there should be at least ``(length - 1)``
        data points before the first ``valid_ind`` in each period, otherwise the
        sequence for that valid_ind will include data from a previous period.

        Args:
            x: Array containing the x variables ordered by period and time
            y: ``None`` or array containing the targets corresponding to the ``x``
                variables
            batch_size: Size of training batches
            valid_inds: Array of ``int`` containing the indices which are valid
                end-points of training sequences (for example, we may set ``valid_inds``
                so it contains only the data points at the start of each hour).
            length: Number of data points in each sequence of the batch
            shuffle: Whether to shuffle ``valid_ind`` before training and after
                each epoch. For training, it is recommended to set this to ``True``, so
                each batch contains a varied sample of data from different times. For
                prediction, it should be set to ``False``, so that the predicted values
                are in the same order as the input data.
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
        """Return the array of labels indexed by ``valid_ind``."""
        if self.y is None:
            raise RuntimeError("Generator has no y data.")
        else:
            return self.y[self.valid_inds]

    def __len__(self):
        """Return the number of batches in each epoch."""
        return int(np.ceil(len(self.valid_inds) / self.batch_size))

    def __getitem__(self, idx):
        """Generate a batch. ``idx`` is the index of the batch in the training epoch."""
        if (idx < self.__len__() - 1) or (len(self.valid_inds) % self.batch_size == 0):
            num_samples = self.batch_size
        else:
            num_samples = len(self.valid_inds) % self.batch_size
        x = np.empty((num_samples, self.length, self.x.shape[1]))
        end_indexes = self.valid_inds[
            idx * self.batch_size : (idx + 1) * self.batch_size
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


def combine_old_and_new_data(
    old_data: pd.DataFrame, new_data: pd.DataFrame
) -> pd.DataFrame:
    """Combine old data at hourly frequency with new data at 1-minute frequency.
    Adjust for different satellite position.

    Args:
        old_data: solar wind data at hourly frequency, measured at bow-shock-nose
        new_data: solar wind data at 1-minute frequency, measured at L1 position

    Returns:
        comb_data: combined data at 1-minute frequency, adjusted to equivalent of
            bow-shock-nose position
    """

    # Average 1-minute data so that the result at time t hours is the average of the
    # 1-minute data between t and t + 59 minutes inclusive.
    # Then shift the data 1 hour forward (data labelled 10:00 is now labelled 11:00).
    new_hourly_arr = []
    new_data["timedelta"] = pd.to_timedelta(new_data["timedelta"])
    if len(old_data) > 0:
        old_data["timedelta"] = pd.to_timedelta(old_data["timedelta"])
    for p in new_data["period"].unique():
        period_data = new_data.loc[new_data["period"] == p].copy()
        period_data_hourly = period_data.resample(
            "1H", closed="left", label="right", on="timedelta", offset="1H"
        ).mean()
        # restore non-numeric columns
        num_num_cols = ["period", "source", "timedelta"]
        period_data_hourly_non_num = (
            period_data[num_num_cols]
            .resample("1H", closed="left", label="right", on="timedelta", offset="1H")
            .first()
        )
        # drop the timedelta column, since the correct timedelta is now the index
        period_data_hourly_non_num.drop("timedelta", inplace=True, axis=1)
        period_data_hourly = pd.concat(
            [period_data_hourly, period_data_hourly_non_num], axis=1
        )
        period_data_hourly.reset_index(inplace=True)
        new_hourly_arr.append(period_data_hourly)

    new_hourly = pd.concat(new_hourly_arr, axis=0, ignore_index=True)
    comb_data = pd.concat([new_hourly, old_data], axis=0, ignore_index=True)
    return comb_data
