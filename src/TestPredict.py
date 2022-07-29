import unittest
import pandas as pd
import os
import numpy as np
import datetime as dt

from predict import load_models, predict_batch, permutation_importance
from sklearn.metrics import mean_squared_error
from plotting import creat_model_uncertainty_array, plot_ErrorBar, plot_permutation_outcome, plot_PeakErrorBar, \
    transform_to_actual_date, plot_final_prediction, plot_prediction_window


class TestPredict(unittest.TestCase):

    def setUp(self) -> None:

        self.data_folder = "data"
        self.output_folder = os.path.join("trained_models", "cnn_1_min", "all_data")
        self.solar_cols = ["period", "timedelta", "bx_gsm", "by_gsm", "bz_gsm", "bt", "speed", "density", "temperature"]

    def test_ErrorBar(self):

        #load test data
        solar_test = pd.read_csv(os.path.join(self.data_folder, "private", "solar_wind.csv"), usecols=self.solar_cols,
                                 dtype={"period": "category"})
        dst_test = pd.read_csv(os.path.join(self.data_folder, "private", "dst_labels.csv"), dtype={"period": "category"})
        sunspots_test = pd.read_csv(os.path.join(self.data_folder, "private", "sunspots.csv"), dtype={"period": "category"})



        #load model
        model_t_arr, model_t_plus_1_arr, norm_df = load_models(self.output_folder, 5)
        dst_test["timedelta"] = pd.to_timedelta(dst_test["timedelta"])
        # exclude times in the first week + 1 hour of dst_test
        dst_test = dst_test.loc[dst_test["timedelta"] >= dt.timedelta(days=7, hours=1)].copy()



       #get prediction
        predictions, t0_predictions_set, t1_predictions_set = predict_batch(
            solar_test.copy(), sunspots_test, dst_test, model_t_arr, model_t_plus_1_arr, norm_df, "minute"
        )

        dst_test_1_min = pd.merge(dst_test, predictions, "left", ["timedelta", "period"])


        #plot ErrorBar
        # plot the limits for the prediction model
        N = len(t0_predictions_set[0][:])

        t0_uncern = creat_model_uncertainty_array(N, t0_predictions_set)


        #loss_t = np.sqrt(
        #    mean_squared_error(dst_test_1_min["dst"].values, dst_test_1_min["prediction_t"].values)
        #)

        #print(f"RMSE for time t: {loss_t:0.2f}")

        start = 1600
        end = 1800
        plot_ErrorBar(start, end, dst_test_1_min, t0_uncern)

    def test_permutation_importance(self):
        # load test data
        solar_test = pd.read_csv(os.path.join(self.data_folder, "private", "solar_wind.csv"), usecols=self.solar_cols,
                                 dtype={"period": "category"})
        dst_test = pd.read_csv(os.path.join(self.data_folder, "private", "dst_labels.csv"),
                               dtype={"period": "category"})
        sunspots_test = pd.read_csv(os.path.join(self.data_folder, "private", "sunspots.csv"),
                                    dtype={"period": "category"})



        # load model
        model_t_arr, model_t_plus_1_arr, norm_df = load_models(self.output_folder, 5)
        dst_test["timedelta"] = pd.to_timedelta(dst_test["timedelta"])
        # exclude times in the first week + 1 hour of dst_test
        dst_test = dst_test.loc[dst_test["timedelta"] >= dt.timedelta(days=7, hours=1)].copy()
        solar_test["timedelta"] = pd.to_timedelta(solar_test["timedelta"])

        permute_cols = ["bx_gsm", "by_gsm", "bz_gsm", "bt", "speed", "density", "temperature"]


        rmse_df = permutation_importance(solar_test.copy(), sunspots_test, dst_test, model_t_arr, model_t_plus_1_arr, norm_df, permute_cols)

        # get prediction
        predictions, t0_predictions_set, t1_predictions_set = predict_batch(
            solar_test.copy(), sunspots_test, dst_test, model_t_arr, model_t_plus_1_arr, norm_df, "minute"
        )

        dst_test_1_min = pd.merge(dst_test, predictions, "left", ["timedelta", "period"])

        loss_t = np.sqrt(
            mean_squared_error(dst_test_1_min["dst"].values, dst_test_1_min["prediction_t"].values)
        )

        print('In order of most important feature first to least important by rmse(j)/rmse:')
        rmse_ratio_df = (rmse_df / loss_t).sort_values(ascending=False, by=0, axis=1)
        print(rmse_ratio_df.T)

        plot_permutation_outcome(rmse_ratio_df)


    def test_see_time_arr_info(self):
        # load test data
        solar_test = pd.read_csv(os.path.join(self.data_folder, "private", "solar_wind.csv"), usecols=self.solar_cols,
                                 dtype={"period": "category"})
        dst_test = pd.read_csv(os.path.join(self.data_folder, "private", "dst_labels.csv"),
                               dtype={"period": "category"})
        sunspots_test = pd.read_csv(os.path.join(self.data_folder, "private", "sunspots.csv"),
                                    dtype={"period": "category"})

        # load model

        dst_test["timedelta"] = pd.to_timedelta(dst_test["timedelta"])
        # exclude times in the first week + 1 hour of dst_test
        dst_test = dst_test.loc[dst_test["timedelta"] >= dt.timedelta(days=7, hours=1)].copy()

        # get prediction
        model_t_arr, model_t_plus_1_arr, norm_df = load_models(self.output_folder, 5)
        predictions, t0_predictions_set, t1_predictions_set = predict_batch(
            solar_test.copy(), sunspots_test, dst_test, model_t_arr, model_t_plus_1_arr, norm_df, "minute"
        )

        dst_test_1_min = pd.merge(dst_test, predictions, "left", ["timedelta", "period"])

        time_arr = dst_test_1_min["timedelta"].astype(str).tolist()

        print(len(dst_test_1_min.loc[dst_test_1_min["period"]=="test_b"]))


        date_arr = transform_to_actual_date(dst_test_1_min)

        print(date_arr[:50])

    def test_dst_label_shift(self):
        # load test data

        dst = pd.read_csv(os.path.join(self.data_folder, "public", "dst_labels.csv"),
                               dtype={"period": "category"})

        solar = pd.read_csv(os.path.join(self.data_folder, "public", "solar_wind.csv"), usecols=self.solar_cols,
                                 dtype={"period": "category"})


        dst["timedelta"] = pd.to_timedelta(dst["timedelta"])
        solar["timedelta"] = pd.to_timedelta(solar["timedelta"])
        solar = pd.merge(solar, dst, "left", ["period", "timedelta"])



        # interpolate target and shift target since we only have data up to t - 1 minute
        solar["target"] = (
            solar["dst"].shift(-1).interpolate(method="linear", limit_direction="both")
        )
        # shift target for training t + 1 hour model
        solar["target_shift"] = solar["dst"].shift(-60)
        solar["target_shift"] = solar["target_shift"].fillna(method="ffill")

        solar["t0"] = solar["timedelta"].shift(-1)
        solar["t1"] = solar["timedelta"].shift(-60)

        print(solar[["target", "t0"]][:10])
        print(solar[["target_shift", "t1"]][:10])




        count = 0

        for t0, t1 in zip(solar["target"], solar["target_shift"]):

            if (t1 - t0) == np.nan:
                print(count)
                print(solar["target"][count])
                print(solar["target_shift"][count])
                break
            count += 1

            self.assertEqual(pd.Timedelta(t1 - t0).seconds / 60, 59)

    def test_plot_all_results(self):

        # load test data
        solar_test = pd.read_csv(os.path.join(self.data_folder, "private", "solar_wind.csv"), usecols=self.solar_cols,
                                 dtype={"period": "category"})
        dst_test = pd.read_csv(os.path.join(self.data_folder, "private", "dst_labels.csv"),
                               dtype={"period": "category"})
        sunspots_test = pd.read_csv(os.path.join(self.data_folder, "private", "sunspots.csv"),
                                    dtype={"period": "category"})

        # load model
        model_t_arr, model_t_plus_1_arr, norm_df = load_models(self.output_folder, 5)
        dst_test["timedelta"] = pd.to_timedelta(dst_test["timedelta"])
        # exclude times in the first week + 1 hour of dst_test
        dst_test = dst_test.loc[dst_test["timedelta"] >= dt.timedelta(days=7, hours=1)].copy()

        # get prediction
        predictions, t0_predictions_set, t1_predictions_set = predict_batch(
            solar_test.copy(), sunspots_test, dst_test, model_t_arr, model_t_plus_1_arr, norm_df, "minute"
        )

        dst_test_1_min = pd.merge(dst_test, predictions, "left", ["timedelta", "period"])

        start = 1600
        end = 1800

        plot_final_prediction(start, end, dst_test_1_min)

    def test_plot_prediction_window(self):

        # load test data
        solar_test = pd.read_csv(os.path.join(self.data_folder, "private", "solar_wind.csv"), usecols=self.solar_cols,
                                 dtype={"period": "category"})
        dst_test = pd.read_csv(os.path.join(self.data_folder, "private", "dst_labels.csv"),
                               dtype={"period": "category"})
        sunspots_test = pd.read_csv(os.path.join(self.data_folder, "private", "sunspots.csv"),
                                    dtype={"period": "category"})

        # load model
        model_t_arr, model_t_plus_1_arr, norm_df = load_models(self.output_folder, 5)
        dst_test["timedelta"] = pd.to_timedelta(dst_test["timedelta"])
        # exclude times in the first week + 1 hour of dst_test
        dst_test = dst_test.loc[dst_test["timedelta"] >= dt.timedelta(days=7, hours=1)].copy()

        # get prediction
        predictions, t0_predictions_set, t1_predictions_set = predict_batch(
            solar_test.copy(), sunspots_test, dst_test, model_t_arr, model_t_plus_1_arr, norm_df, "minute"
        )

        dst_test_1_min = pd.merge(dst_test, predictions, "left", ["timedelta", "period"])

        plot_prediction_window(dst_test, dst_test_1_min)
















