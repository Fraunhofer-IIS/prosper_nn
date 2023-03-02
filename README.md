# Prosper

We present the package prosper\_nn, that provides four neural network architectures dedicated to time series forecasting, implemented in PyTorch: the Error Correction Neural Network (ECNN), the Historical Consistent Neural Network (HCNN), the Causal-Retro-Causal Neural Network (CRCNN), and the Fuzzy Neural Network. In addition, prosper\_nn contains the first sensitivity analysis suitable for RNNs and a heatmap to visualize forecasting uncertainty which was previously only available in Java.
These models and methods are used in industry for two decades and were used and referenced in several scientific publications. However, only now we make them publicly available, allowing researchers and practitioners to benchmark and further develop them.	The package is designed to make the models easily accessible, thereby enabling research and application.
The full documentation can be found on https://fraunhofer-iis.github.io/prosper_nn/. There are also tutorials that show how to work with the package.

This work was supported by the ADA Lovelace Center for Analytics, Data and Applications.

## Installation

You can use pip to install the package:

`pip install prosper-nn`

Afterwards you can import prosper_nn with:

`import prosper_nn`
