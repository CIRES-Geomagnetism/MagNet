import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


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
