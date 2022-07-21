import datetime
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import datetime as dt


def plot_binned_RMSE(actual: np.ndarray, predicted: np.ndarray, bin_edges):
    """Produce a plot of RMSE, binned by dst range.

    Args:
        actual: true values of dst
        predicted: predicted values of dst
        bin_edges: edges of the bins by which to group the data
    """

    df = pd.DataFrame({"actual": actual, "predicted": predicted})
    df["bin"] = pd.cut(df["actual"], bin_edges)
    df["sq_err"] = (df["actual"] - df["predicted"]) ** 2
    RMSE_by_bin = np.sqrt(df.groupby("bin")["sq_err"].mean())
    plt.plot(RMSE_by_bin.values, marker='.', markersize=15)
    labels = [s.replace(",", ",\n") for s in RMSE_by_bin.index.astype(str).values]
    plt.xticks(ticks=np.arange(len(RMSE_by_bin)), labels=labels)
    plt.xlabel("DST range")
    plt.ylabel("RMSE")


def creat_model_uncertainty_array(N: int, predictions_set: List):

    uncertainty = []
    M = len(predictions_set)
    time_steps = [0] * M

    for j in range(N):
        for i in range(M):
            time_steps[i] = predictions_set[i][j]
        max_val = max(time_steps)
        min_val = min(time_steps)
        uncertainty.append(max_val - min_val)

    return uncertainty

def select_by_freq(true_dst: np.ndarray, predict_dst: np.ndarray, uncertainty: List, freq: int) -> Tuple[np.ndarray, np.ndarray,  np.ndarray,  np.ndarray]:

    time = []
    true = []
    pred = []
    predErr = []

    i = 0
    while i < len(true_dst):
        time.append(i)
        true.append(true_dst[i])
        pred.append(predict_dst[i])
        predErr.append(uncertainty[i])
        i += freq

    time = np.array(time, dtype=float)
    true = np.array(true, dtype=float)
    pred = np.array(pred, dtype=float)
    predErr = np.array(predErr, dtype=float)

    return time, true, pred, predErr



# define the function to plot Errorbar for each model
def plot_ErrorBar(start: int, end: int, dst_test_1_min: pd.DataFrame, predErr: List):
    """
    Plot the Error Bar
    Args:
        start: start hout
        end: end hout
        dst_test_1_min: Dataframe include Dst test data and Dst predicted data
        predErr: the max and min of predicted Dst in each time steps
    """

    true = dst_test_1_min["dst"].values
    pred = dst_test_1_min["prediction_t"].values
    dates = transform_to_datetime(dst_test_1_min)

    fig, ax = plt.subplots()
    plt.rcParams["figure.figsize"] = (12, 9)

    ax.errorbar(dates[start:end], pred[start:end], predErr[start:end], color="b", label="limits of predicted Dst", elinewidth = 2)
    ax.plot(dates[start:end], true[start:end], color="r", label="true data")

    # Tell matplotlib to interpret the x-axis values as dates
    ax.xaxis_date()
    # Make space for and rotate the x-axis tick labels
    fig.autofmt_xdate()

    ax.set(xlabel="Date", ylabel="Dst", title="Uncertainty Quantifucation(UQ) of the Ensamble CNN Model")
    plt.legend(loc='lower right')
    plt.show()

def plot_permutation_outcome(rmse_ratio_df: pd.DataFrame):

    plt.plot(rmse_ratio_df.columns, rmse_ratio_df.values.T, 'x-')
    plt.xticks(rotation=270)
    plt.ylabel('RMSE Ratio %')
    plt.grid(True)
    plt.show()

def plot_PeakErrorBar(true_dst: np.ndarray, predict_dst: np.ndarray, uncertainty: List, offset: int):

    plt.figure(figsize=(16, 12))

    time, true, pred, predErr = select_by_freq(true_dst, predict_dst, uncertainty, offset)

    idx = np.argmin(true)

    print(true[idx])

    peak_start = idx - 100
    peak_end = idx + 100


    plt.errorbar(time[peak_start:peak_end], pred[peak_start:peak_end], predErr[peak_start:peak_end], color="b", label="limits of predicted Dst", elinewidth = 2)
    plt.plot(time[peak_start:peak_end], true[peak_start:peak_end], color="r", label="true data")

    plt.title("Uncertainty Quantification(UQ) of the Ensamble CNN Model")
    plt.legend(loc='lower right')
    plt.xlabel("Date")
    plt.ylabel("Dst")
    plt.show()

def transform_to_actual_date(df: pd.DataFrame) -> List:
    """
    Args:
        df: Dataframe
    Returns:
        all_time: array for the datetime in string type
    """
    start_a = datetime.datetime.strptime("2001-06-01 00:00:00", "%Y-%m-%d %H:%M:%S")
    start_b = datetime.datetime.strptime("2011-01-01 00:00:00", "%Y-%m-%d %H:%M:%S")
    start_c = datetime.datetime.strptime("2019-06-01 00:00:00", "%Y-%m-%d %H:%M:%S")


    range_a = len(df.loc[df["period"] == "test_a"])
    range_b = len(df.loc[df["period"] == "test_b"])
    range_c = len(df.loc[df["period"] == "test_c"])

    all_time = [(start_a + datetime.timedelta(hours=1 * i)).strftime("%m/%d/%Y %H:%M") for i in range(range_a)]
    time_b = [(start_b + datetime.timedelta(hours=1 * i)).strftime("%m/%d/%Y %H:%M") for i in range(range_b)]
    time_c = [(start_c + datetime.timedelta(hours=1 * i)).strftime("%m/%d/%Y %H:%M") for i in range(range_c)]

    all_time.extend(time_b)
    all_time.extend(time_c)


    return all_time

def transform_to_datetime(df: pd.DataFrame):
    """
    Args:
        df: pandas Dataframe

    Returns:
        all_times: the datetime array with all of the test data
    """

    start_a = datetime.datetime.strptime("2001-06-01 00:00", "%Y-%m-%d %H:%M")
    start_b = datetime.datetime.strptime("2011-01-01 00:00", "%Y-%m-%d %H:%M")
    start_c = datetime.datetime.strptime("2019-06-01 00:00", "%Y-%m-%d %H:%M")

    range_a = len(df.loc[df["period"] == "test_a"])
    range_b = len(df.loc[df["period"] == "test_b"])
    range_c = len(df.loc[df["period"] == "test_c"])

    all_time = [(start_a + datetime.timedelta(hours=1 * i)) for i in range(range_a)]
    time_b = [(start_b + datetime.timedelta(hours=1 * i)) for i in range(range_b)]
    time_c = [(start_c + datetime.timedelta(hours=1 * i)) for i in range(range_c)]

    all_time.extend(time_b)
    all_time.extend(time_c)

    return all_time


def plot_final_prediction(start:int, end:int, dst: pd.DataFrame):
    """
    Plot the prediction of t0 and t1
    Args:
        start: start hour
        end: end hour
        dst: Dataframe of Dst
    """

    true = dst["dst"].values
    pred_t0 = dst["prediction_t"].values
    pred_t1 = dst["prediction_t_plus_1"].values

    dates = transform_to_datetime(dst)
    adjust_t1 = np.concatenate(([np.nan], pred_t1[:-1]))
    dates, true, pred_t0, adjust_t1 = dates[start:end], true[start:end], pred_t0[start:end], adjust_t1[start:end]

    fig, ax = plt.subplots()
    plt.rcParams["figure.figsize"] = (12, 9)

    ax.plot(dates, true, label="Dst Observed")
    ax.scatter(dates, pred_t0, color='b', s=1, label="Dst Prediction t0")
    ax.scatter(dates, adjust_t1, color='r', s=1, label="Dst Prediction t1")

    ax.set(xlabel="Date", ylabel="Dst", title=f"Dst Prediction Results from {dates[0]} to {dates[-1]}")
    plt.legend(loc='lower right')

    # Tell matplotlib to interpret the x-axis values as dates
    ax.xaxis_date()
    # Make space for and rotate the x-axis tick labels
    fig.autofmt_xdate()

    plt.show()

def plot_prediction_window(dst_test: pd.DataFrame, dst_test_1_min: pd.DataFrame):
    """
    Plot the prediction of t0 in different time range
    Args:
        dst_test: Dataframe only contain Dst test data
        dst_test_1_min: Dataframe include Dst test data and predicted data
    """
    # select events at least 7 days apart
    num_events = 10
    min_ind_arr = []
    min_time_arr = []
    # dst
    dst_test['exclude'] = False
    for i in range(num_events):
        min_ind = dst_test.loc[~dst_test['exclude'], 'dst'].idxmin()
        min_time = dst_test.loc[min_ind, ['timedelta', 'period']]
        min_time_arr.append(min_time)
        t, p = min_time['timedelta'], min_time['period']
        min_ind_arr.append(min_ind)
        dst_test['exclude'] = dst_test['exclude'] | (
                    ((dst_test['timedelta'] - t).dt.total_seconds().abs() <= 7 * 24 * 3600) & (dst_test['period'] == p))

    # sort by period and timedelta
    sort_ind = list(range(num_events))
    sort_ind = sorted(sort_ind, key=lambda x: (min_time_arr[x]['period'], min_time_arr[x]['timedelta']))
    min_ind_arr = [min_ind_arr[i] for i in sort_ind]

    dates = transform_to_datetime(dst_test_1_min)

    for i in range(num_events):
        fig, ax = plt.subplots()
        # extract 96 hours before and after max
        ind = min_ind_arr[i]
        # centre on min within 96 * 2 hour window
        new_min = dst_test.loc[ind - 96: ind + 96, 'dst'].idxmin()
        df = dst_test.loc[new_min - 96: new_min + 96].copy()
        df = pd.merge(df, dst_test_1_min[["period", "timedelta", "prediction_t"]], how="left",
                      on=["timedelta", "period"])

        curr_dates = dates[new_min - 96: new_min + 96 + 1]

        plt.rcParams["figure.figsize"] = (12, 9)
        ax.plot(curr_dates, df["dst"].values, c="black", label="Dst")
        ax.plot(curr_dates, df["prediction_t"].values, c="blue", label="prediction_t")

        # Tell matplotlib to interpret the x-axis values as dates
        ax.xaxis_date()
        # Make space for and rotate the x-axis tick labels
        fig.autofmt_xdate()

        ax.set_title(f"Dst Prediction from {dates[new_min - 96]} to {dates[new_min + 96]}")
        plt.ylabel("Dst")
        plt.show()






