# MagNet

### What is MagNet?
MagNet used convolutional neural network to 
forecast the space weather disturbance-storm-time (Dst) index. 
This model is developed by Dr. Belinda Trotta who wins the second-place solution 
to the [MagNet Competition](https://www.drivendata.org/competitions/73/noaa-magnetic-forecasting/) held by 
the National Oceanic and Atmospheric Administration (NOAA) and National Aeronautics and 
Space Administration (NASA) in 2020-21.
The new [real-time High Definition Geomagnetic Model](https://www.ngdc.noaa.gov/geomag/HDGM/hdgm_rt.html) 
will be driven in part by MagNet model. (It will be released at Nov. 2022.)

The notebooks: [tutorial/magnet_cnn_tutorial.ipynb](https://github.com/liyo6397/MagNet/blob/explainable_ai/tutorials/magnet_cnn_tutorial.ipynb) 
describes the machine learning approach to forecast the Dst index. It was selected 
as the one of the materials for [Trustworthy Artificial Intelligence for Environmental Science(TAI4ES) 2022 
Summer School](https://www2.cisl.ucar.edu/events/tai4es-2022-summer-school). 
### Prerequirements:
- `tensorflow == 2.7.0`
- `numpy == 1.22.1`
- `mysql-connector-python == 8.0.27`
- `pandas == 1.3.5`
- `protobuf==3.20.1`


### How to use MagNet?

The repository contains training and prediction code for the second-place model in the MagNet competition.
There is also code for defining and benchmarking some other models.

To benchmark model performance, using the competition's private data as a test set, run `benchmark.py`.
To train the model on the public and private datasets, run `train_on_all_data.py`.
Prediction functions are in `predict.py`.


- [doc](https://github.com/liyo6397/MagNet/tree/master/doc): The documentation for the architecture of MagNet CNN models.
- [tutorials](https://github.com/liyo6397/MagNet/tree/explainable_ai/tutorials): It contains the tutorials for the machine learning models which has won first
and second place in MagNet competition. To use the tutorial for MagNet CNN codes, please follows: [tutorial/magnet_cnn_tutorial.ipynb](https://github.com/liyo6397/MagNet/blob/explainable_ai/tutorials/magnet_cnn_tutorial.ipynb).
- [src](https://github.com/liyo6397/MagNet/tree/explainable_ai/src): The source codes for training and benchmarking DST prediction models.