import os
from typing import Callable, List, Optional, Tuple, Union
import random
import inspect
import pandas as pd
import numpy as np

# uncomment to turn off gpu (see https://stackoverflow.com/a/45773574)
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import tensorflow as tf

from preprocessing import (
    prepare_data_1_min,
    prepare_data_hourly,
    prepare_data_hybrid,
    DataGen,
)
from predict import load_models


def train_on_prepared_data(
    prepared_data: pd.DataFrame,
    model: tf.keras.Model,
    initial_weights: Union[List[List[np.ndarray]], List[np.ndarray]],
    epochs: int,
    lr: float,
    bs: int,
    train_cols,
    num_models: int = 1,
    output_folder: str = "trained_models",
    data_frequency: str = "minute",
    early_stopping: bool = False,
    comb_model: bool = False
) -> Optional[List[float]]:
    """Train and save ensemble of models, each trained on a different subset of data.

    Args:
        prepared_data: DataFrame containing solar wind data, sunspots data, and labels
        model: keras neural network model,
        initial_weights: initial weights for the model. Either a single set of weights
            to use for all models, or a list of 2 * num_models sets of weights,
            representing weights for the t models followed by the t + 1 models.
        epochs: number of training epochs,
        lr: learning rate,
        bs: batch size,
        train_cols: columns of ``prepared_data`` to use for training. Must match the
            input size defined in ``model_definer``.
        num_models: Number of models to train. See details under definition of
            ``train_nn_models``.
        output_folder: Path to the directory where models will be saved
        data_frequency: frequency of the training data: "minute" or "hour"
        early_stopping: If ``True``, stop model training when validation loss stops
            decreasing.  See details under definition of ``train_nn_models``.
        comb_model: If ``True``, train single model for times ``t`` and ``t+1``.

    Returns:
        out-of-sample accuracy: If ``num_models > 1``, returns list of length
            ``num_models`` containing RMSE values for out-of-sample predictions for each
            model.  See details under definition of ``train_nn_models``.
    """

    # define model and training parameters
    if data_frequency == "minute":
        sequence_length = 6 * 24 * 7
    else:
        sequence_length = 24 * 7

    # set seeds
    np.random.seed(0)
    random.seed(0)
    tf.random.set_seed(0)

    oos_accuracy = []
    # train on sequences ending at the start of an hour
    valid_bool = prepared_data["timedelta"].dt.seconds % 3600 == 0
    # exclude periods where data contains nans
    nans_in_train = (
        prepared_data[train_cols]
        .isnull()
        .any(axis=1)
        .rolling(window=sequence_length + 1, center=False)
        .max()
    )
    valid_bool = valid_bool & (nans_in_train == 0)
    np.random.seed(0)
    prepared_data["month"] = prepared_data["month"].astype(int)
    months = np.sort(prepared_data["month"].unique())
    # remove the first week from each period, because not enough data for prediction
    valid_ind_arr = []
    for p in prepared_data["period"].unique():
        all_p = prepared_data.loc[
            (prepared_data["period"] == p) & valid_bool
        ].index.values[24 * 7 :]
        valid_ind_arr.append(all_p)
    valid_ind = np.concatenate(valid_ind_arr)
    non_exclude_ind = prepared_data.loc[
        ~prepared_data["train_exclude"].astype(bool)
    ].index.values
    np.random.shuffle(months)
    es_callback = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=5, restore_best_weights=True
    )
    lr_callback = tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss",
                                                       factor=0.2,
                                                       patience=3,
                                                       min_lr=1e-6,
                                                       mode="min")
    callbacks = []
    if early_stopping:
        callbacks.append(es_callback)
        epochs = 100
    for model_ind in range(num_models):
        if isinstance(initial_weights[0], np.ndarray):
            initial_weights_t = initial_weights
            initial_weights_t_plus_1 = initial_weights
        else:
            initial_weights_t = initial_weights[model_ind]
            initial_weights_t_plus_1 = initial_weights[num_models + model_ind]
        # t model
        tf.keras.backend.clear_session()
        model.compile(
            loss=tf.keras.losses.MeanSquaredError(),
            optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
            metrics=[tf.keras.metrics.RootMeanSquaredError()],
        )
        model.set_weights(initial_weights_t)
        if num_models > 1:
            # define train and test sets
            # for p in prepared_data["period"].unique():
            #     curr_months = list(np.sort(np.unique(prepared_data.loc[prepared_data["period"] == p, "month"])))
            #     leave_out_months += curr_months[
            #         model_ind
            #         * (len(curr_months) // num_models) : (model_ind + 1)
            #         * (len(curr_months) // num_models)
            #     ]
            leave_out_months = months[
                model_ind
                * (len(months) // num_models) : (model_ind + 1)
                * (len(months) // num_models)
            ]
            leave_out_months_ind = prepared_data.loc[
                valid_bool & prepared_data["month"].isin(leave_out_months)
            ].index.values
            curr_months_ind = prepared_data.loc[
                valid_bool & (~prepared_data["month"].isin(leave_out_months))
            ].index.values
            train_ind = np.intersect1d(
                np.intersect1d(valid_ind, curr_months_ind), non_exclude_ind
            )
            test_ind = np.intersect1d(valid_ind, leave_out_months_ind)
            if comb_model:
                train_gen = DataGen(
                    prepared_data[train_cols].values,
                    train_ind,
                    prepared_data[["target", "target_shift"]].values,
                    bs,
                    sequence_length,
                )
                test_gen = DataGen(
                    prepared_data[train_cols].values,
                    test_ind,
                    prepared_data[["target", "target_shift"]].values,
                    bs,
                    sequence_length,
                )
            else:
                train_gen = DataGen(
                    prepared_data[train_cols].values,
                    train_ind,
                    prepared_data["target"].values.flatten(),
                    bs,
                    sequence_length,
                )
                test_gen = DataGen(
                    prepared_data[train_cols].values,
                    test_ind,
                    prepared_data["target"].values.flatten(),
                    bs,
                    sequence_length,
                )
            model.fit(
                train_gen,
                validation_data=test_gen,
                epochs=epochs,
                verbose=1,
                callbacks=callbacks,
            )
            oos_accuracy.append(model.evaluate(test_gen, verbose=2)[1])
            print("Out of sample accuracy: ", oos_accuracy)
            print("Out of sample accuracy mean: {}".format(np.mean(oos_accuracy)))
        else:
            if early_stopping:
                leave_out_months = []
                for p in prepared_data["period"].unique():
                    curr_months = list(np.sort(np.unique(prepared_data.loc[prepared_data["period"] == p, "month"])))
                    # leave_out_months += curr_months[
                    #     model_ind
                    #     * (len(curr_months) // 5) : (model_ind + 1)
                    #     * (len(curr_months) // 5)
                    # ]
                leave_out_months = months[
                    model_ind
                    * (len(months) // 5) : (model_ind + 1)
                    * (len(months) // 5)
                ]
                leave_out_months_ind = prepared_data.loc[
                    valid_bool & prepared_data["month"].isin(leave_out_months)
                ].index.values
                curr_months_ind = prepared_data.loc[
                    valid_bool & (~prepared_data["month"].isin(leave_out_months))
                ].index.values
                train_ind = np.intersect1d(
                    np.intersect1d(valid_ind, curr_months_ind), non_exclude_ind
                )
                test_ind = np.intersect1d(valid_ind, leave_out_months_ind)
                if comb_model:
                    train_gen = DataGen(
                        prepared_data[train_cols].values,
                        train_ind,
                        prepared_data[["target", "target_shift"]].values,
                        bs,
                        sequence_length,
                    )
                    test_gen = DataGen(
                        prepared_data[train_cols].values,
                        test_ind,
                        prepared_data[["target", "target_shift"]].values,
                        bs,
                        sequence_length,
                    )
                else:
                    train_gen = DataGen(
                        prepared_data[train_cols].values,
                        train_ind,
                        prepared_data["target"].values.flatten(),
                        bs,
                        sequence_length,
                    )
                    test_gen = DataGen(
                        prepared_data[train_cols].values,
                        test_ind,
                        prepared_data["target"].values.flatten(),
                        bs,
                        sequence_length,
                    )
                model.fit(
                    train_gen,
                    validation_data=test_gen,
                    epochs=epochs,
                    verbose=1,
                    callbacks=callbacks,
                )
            else:
                # fit on all data
                train_ind = valid_ind
                train_gen = DataGen(
                    prepared_data[train_cols].values,
                    train_ind,
                    prepared_data["target"].values.flatten(),
                    bs,
                    sequence_length,
                )
                model.fit(train_gen, epochs=epochs, verbose=1, callbacks=callbacks)
        model.save(os.path.join(output_folder, "model_t_{}.h5".format(model_ind)))
        if early_stopping:
            with open(os.path.join(output_folder, "log.txt"), "a") as f:
                es_iter = es_callback.stopped_epoch - es_callback.patience + 1
                f.write(f"\n\nEarly stopping iterations: {es_iter}")
        if not (comb_model):
            # t + 1 model
            tf.keras.backend.clear_session()
            model.compile(
                loss=tf.keras.losses.MeanSquaredError(),
                optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                metrics=[tf.keras.metrics.RootMeanSquaredError()],
            )
            model.set_weights(initial_weights_t_plus_1)
            data_gen = DataGen(
                prepared_data[train_cols].values,
                train_ind,
                prepared_data["target_shift"].values.flatten(),
                bs,
                sequence_length,
            )
            if early_stopping and (es_callback.stopped_epoch > 0):
                model.fit(
                    data_gen,
                    epochs=es_callback.stopped_epoch - es_callback.patience + 1,
                    verbose=1,
                )
            else:
                model.fit(data_gen, epochs=epochs, verbose=1)
            model.save(
                os.path.join(output_folder, "model_t_plus_one_{}.h5".format(model_ind))
            )
    if num_models > 1:
        return oos_accuracy


def train_nn_models(
    solar: pd.DataFrame,
    sunspots: pd.DataFrame,
    dst: pd.DataFrame,
    model_definer: Callable[
        [], Tuple[tf.keras.Model, List[np.ndarray], int, float, int]
    ],
    num_models: int = 1,
    output_folder: str = "trained_models",
    data_frequency: str = "minute",
    early_stopping: bool = False,
) -> Optional[List[float]]:
    """Train and save ensemble of models, each trained on a different subset of data.

    Args:
        solar: DataFrame containing solar wind data
        sunspots: DataFrame containing sunspots data
        dst: DataFrame containing the disturbance storm time (DST) data, i.e. the labels
            for training
        model_definer: Function returning keras model and initial weights
        num_models: Number of models to train.
            Training several models on different subsets of the data and averaging
            results improves model accuracy. If ``num_models = 1``, use all data for
            training. If ``num_models > 1``, each model will be trained on
            ``int((num_models - 1) / num_models) * num_months`` randomly-selected months
            of data, where ``num_months`` is the total number of months in the training
            data. Because the data contains long-term trends lasting days or weeks,
            splitting the data by month ensures that different models have sufficiently
            different data sets to obtain the benefits of model diversity. Separate
            models are trained for the current and next hour, so the number of models
            output will be ``num_models * 2``.
        output_folder: Path to the directory where models will be saved
        data_frequency: frequency of the training data: "minute" or "hour"
        early_stopping: If ``True``, stop model training when validation loss stops
            decreasing. If ``num_models > 1``, use out-of-fold data as the validation
            set; otherwise randomly select 20% of months. Early stopping is used
            only on the time ``t`` model, and the resulting optimal number of epochs is
            used to train the ``t + 1`` model.

    Returns:
        out-of-sample accuracy: If ``num_models > 1``, returns list of length
            ``num_models`` containing RMSE values for out-of-sample predictions for each
            model. These provide an indication of model accuracy, but in general will
            underestimate the accuracy of the full ensemble of models. If
            ``num_models = 1``, returns ``None``.
    """

    # prepare data
    if data_frequency == "minute":
        solar, train_cols = prepare_data_1_min(
            solar, sunspots, dst, output_folder=output_folder, norm_df=None
        )
    else:
        solar, train_cols = prepare_data_hourly(
            solar, sunspots, dst, output_folder=output_folder, norm_df=None
        )
    model, initial_weights, epochs, lr, bs = model_definer()

    # Write model definition to log file
    with open(os.path.join(output_folder, "log.txt"), "w") as f:
        f.write(inspect.getsource(model_definer))

    return train_on_prepared_data(
        solar,
        model,
        initial_weights,
        epochs,
        lr,
        bs,
        train_cols,
        num_models,
        output_folder,
        data_frequency,
        early_stopping,
    )


def train_nn_hybrid_models(
    solar_hourly: pd.DataFrame,
    solar_1_min: pd.DataFrame,
    sunspots: pd.DataFrame,
    dst: pd.DataFrame,
    model_definer: Callable[
        [],
        Tuple[
            Tuple[tf.keras.Model, List[np.ndarray], int, float, int],
            Tuple[tf.keras.Model, List[np.ndarray], int, float, int],
        ],
    ],
    num_models: int = 1,
    output_folder: str = "trained_models",
    output_folder_hourly: str = "trained_models",
    early_stopping: bool = False,
    freeze_hourly_layers: bool = True,
) -> Optional[List[float]]:
    """Train and save ensemble of models, each trained on a different subset of data.
    This function is similar to  ``train_nn_models``, but is for training hybrid
    models that use both hourly and 1-minute data.

    Args:
        solar_hourly: DataFrame containing hourly solar wind data
        solar_1_min: DataFrame containing 1-minute data
        sunspots: DataFrame containing sunspots data
        dst: DataFrame containing the disturbance storm time (DST) data, i.e. the labels
            for training
        model_definer: Function returning keras model and initial weights. Layers
            for the hourly data must have name beginning with ``hour``.
        num_models: Number of models to train.
            Training several models on different subsets of the data and averaging
            results improves model accuracy. If ``num_models = 1``, use all data for
            training. If ``num_models > 1``, each model will be trained on
            ``int((num_models - 1) / num_models) * num_months`` randomly-selected months
            of data, where ``num_months`` is the total number of months in the training
            data. Because the data contains long-term trends lasting days or weeks,
            splitting the data by month ensures that different models have sufficiently
            different data sets to obtain the benefits of model diversity. Separate
            models are trained for the current and next hour, so the number of models
            output will be ``num_models * 2``.
        output_folder: Path to the directory where models will be saved
        output_folder_hourly: Path to the directory to save the hourly parts of models
        early_stopping: If ``True``, stop model training when validation loss stops
            decreasing. If ``num_models > 1``, use out-of-fold data as the validation
            set; otherwise randomly select 20% of months. Early stopping is used
            only on the time ``t`` model, and the resulting optimal number of epochs is
            used to train the ``t + 1`` model.
        freeze_hourly_layers: If ``True`` freeze weights of hourly layers before
            fine-tuning on 1-minute data.

    Returns:
        out-of-sample accuracy: If ``num_models > 1``, returns list of length
            ``num_models`` containing RMSE values for out-of-sample predictions for each
            model. These provide an indication of model accuracy, but in general will
            underestimate the accuracy of the full ensemble of models. If
            ``num_models = 1``, returns ``None``.
    """

    # prepare data
    (
        solar_1_min,
        solar_hourly,
        train_cols_1_min,
        train_cols_hourly,
    ) = prepare_data_hybrid(
        solar_1_min,
        solar_hourly,
        sunspots,
        dst,
        None,
        output_folder,
        output_folder_hourly,
        freq_for_1_min_data="10_minute"
    )

    # define model and training parameters
    (
        (minute_model, initial_weights_minute, minute_epochs, minute_lr, minute_bs),
        (hour_model, initial_weights_hour, hour_epochs, hour_lr, hour_bs),
    ) = model_definer()

    # train the hourly part
    train_on_prepared_data(
        solar_hourly,
        hour_model,
        initial_weights_hour,
        hour_epochs,
        hour_lr,
        hour_bs,
        train_cols_hourly,
        num_models,
        output_folder=output_folder_hourly,
        data_frequency="hour",
        early_stopping=early_stopping,
    )

    # copy weights from hourly model over hourly layers of initial_weights
    model_t_arr, model_t_plus_1_arr, _ = load_models(output_folder_hourly, num_models)
    hour_to_minute_layer_dict = {}
    for i, layer in enumerate(hour_model.layers):
        if "hour" in layer.name:
            j = 0
            while j < len(minute_model.layers):
                if minute_model.layers[j].name == layer.name:
                    hour_to_minute_layer_dict[i] = j
                    break
                j += 1
    minute_weights_arr = []
    # t models
    for model_ind in range(num_models):
        for i in range(len(model_t_arr[model_ind].layers)):
            if i in hour_to_minute_layer_dict:
                minute_layer_ind = hour_to_minute_layer_dict[i]
                minute_model.layers[minute_layer_ind].set_weights(
                    model_t_arr[model_ind].layers[i].get_weights()
                )
        minute_weights_arr.append(minute_model.get_weights())
    # t + 1 models
    for model_ind in range(num_models):
        for i in range(len(model_t_plus_1_arr[model_ind].layers)):
            if i in hour_to_minute_layer_dict:
                minute_layer_ind = hour_to_minute_layer_dict[i]
                minute_model.layers[minute_layer_ind].set_weights(
                    model_t_plus_1_arr[model_ind].layers[i].get_weights()
                )
        minute_weights_arr.append(minute_model.get_weights())

    if freeze_hourly_layers:
        for layer in minute_model.layers:
            if "hour" in layer.name:
                layer.trainable = False

    # train the complete model
    oos_accuracy = train_on_prepared_data(
        solar_1_min,
        minute_model,
        minute_weights_arr,
        minute_epochs,
        minute_lr,
        minute_bs,
        train_cols_1_min,
        num_models,
        output_folder=output_folder,
        data_frequency="minute",
        early_stopping=early_stopping,
    )
    return oos_accuracy
