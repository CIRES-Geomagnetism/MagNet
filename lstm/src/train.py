import tensorflow as tf

from tensorflow.keras.layers import Dense, LSTM,GRU, Bidirectional, Dropout, Embedding, Input, Flatten
from tensorflow.keras.models import Sequential
import keras
from tensorflow.keras.layers import Dense, LSTM,GRU, Bidirectional, Dropout, GlobalAveragePooling1D,Input,Concatenate,Flatten,Embedding,Reshape,Conv1D,TimeDistributed,BatchNormalization,GaussianNoise
from tensorflow.keras.models import Sequential

class model:

    def __init__(self, data_config: dict, training_cols: list):

        self.data_config = data_config


        # all of the features we'll use, including sunspot numbers
        self.XCOLS = training_cols
    def define_model(self):
        # define dummy model
        model_config = {"n_epochs": 100, "n_neurons": 244 * 2, "dropout": 0.0, "stateful": False}

        input1 = Input(shape=(self.data_config["timesteps"], len(self.XCOLS)), name='x1')
        lstm1 = Bidirectional(LSTM(
            model_config["n_neurons"],
            stateful=model_config["stateful"],
            dropout=model_config["dropout"], return_sequences=True
        ))(input1)
        gru1 = Bidirectional(GRU(
            model_config["n_neurons"] * 3,
            stateful=model_config["stateful"],
            dropout=0.0, return_sequences=True
        ))(lstm1)

        gaverage = Flatten()(gru1)
        dense1 = Dense(96)(gaverage)
        dense1 = Dense(128)(dense1)
        dense1 = Dense(64)(dense1)
        dense = Dense(2)(dense1)

        model = keras.models.Model(inputs=input1, outputs=dense)
        model.compile(
            loss='mean_squared_error',
            optimizer=tf.keras.optimizers.Adam(0.0001),
        )

        model.summary()

