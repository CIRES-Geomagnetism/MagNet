import pandas as pd
import os
import datetime as dt

import preprocessing as pps
import predict

def main():

    model_folder = "trained_model"
    # load dataframes from csv files
    data_folder = "data"
    solar_cols = ["period", "timedelta", "bx_gsm", "by_gsm", "bz_gsm", "bt", "speed", "density", "temperature"]

    solar_test = pd.read_csv(os.path.join(data_folder, "private", "solar_wind.csv"), usecols=solar_cols,
                             dtype={"period": "category"})
    dst_test = pd.read_csv(os.path.join(data_folder, "private", "dst_labels.csv"), dtype={"period": "category"})
    sunspots_test = pd.read_csv(os.path.join(data_folder, "private", "sunspots.csv"), dtype={"period": "category"})

    model_t_arr, model_t_plus_1_arr, norm_df = predict.load_models(model_folder, 1)
    dst_test["timedelta"] = pd.to_timedelta(dst_test["timedelta"])
    # exclude times in the first week + 1 hour of dst_test
    dst_test = dst_test.loc[dst_test["timedelta"] >= dt.timedelta(days=7, hours=1)].copy()

if __name__ == "__main__":
    main()
