Quick start guide
=================

The repository contains training and prediction code for the second-place model in the
MagNet competition.

There is also code for defining and benchmarking some other models.

The code requires tensorflow and some common numerical computation packages (e.g. pandas, numpy). Apart from tensorflow,
all required packages are part of the standard Anaconda distribution. Training takes around 15 minutes on the CPU.

To benchmark model performance, using the competition's private data as a test set, run ``benchmark.py``.

To train the model on the public and private datasets, run ``train_on_all_data.py``.

Prediction functions are in ``predict.py``.
