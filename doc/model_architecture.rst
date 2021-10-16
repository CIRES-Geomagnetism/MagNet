Model architecture
==================

Here we describe the model architecture for the CNN model defined in ``model_definitions.define_model_cnn``.

.. figure:: _static/cnn_diagram.pdf
   :scale: 90 %

   CNN model architecture.

The model consists of a set of convolutional layers which detect patterns at progressively longer time spans. At each convolutional
layer (except the last), a convolutional filter is applied having size 6 with stride 3, which reduces the size of the output data
relative to the input. (The last convolution has small input size, so it just convolves all its 9 inputs
together.) Thee earlier layers recognize low-level features on short time-spans, and these are outputs aggregated into higher-level
patterns spanning longer time ranges in the later layers. Cropping is applied at each layer which removes a few data points at the beginning at the
sequence to ensure the result will be exactly divisible by 6, so that the last application of the convultional filter will
capture the data at the very end of the sequence. Following all the convolutional
layers is a layer which concatenates the last data point of each of the convolution outputs. This concatenation is then fed into a dense layer. The idea of taking the last data point of each convolution
is that it represents the patterns  at different timespans leading up to the prediction time: for example, the last data point
of the first layer gives the features of the hour before the prediction time, then the second layer gives
the last 6 hours, etc.

The architecture is loosely inspired by a widely used architecture for image
segmentation, the U-Net introduced by Ronneberger, Fischer, and Brox. The idea
is adapted to work for 1-D sequence data instead of 2-D image data, and to predict
only the one value at the end of the sequence (rather than every point, as in image
segmentation). The U-Net consists of a "contracting path", a series of
convolutional layers which condense the image, followed by an "expansive path"
of up-convolution layers which expand the outputs back to the scale of the original
image. At each step of the expansive path, the up-convolution provides information
about the surrounding large-scale patterns, which is combined with to the output
of the corresponding step of the contracting path which provides information about
the small-scale local detail. Combining small-scale and large-scale features
allows the network to make localised predictions that also take account of
larger surrounding patterns.

Similarly, the this model is designed to combine small-scale and large-scale
patterns. Concatenating the outputs from earlier layers captures the small-scale
detail near the end of the sequence, while the output of the final convolutional
layer provides information about  larger-scale patterns. The model uses only the
last output value from each layer, because (unlike U-Net) it only predicts one
value at the end of the sequence, so it suffices to capture local detail near
that point.
