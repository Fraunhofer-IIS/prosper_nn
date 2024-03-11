[![DOI](https://zenodo.org/badge/589583113.svg)](https://zenodo.org/doi/10.5281/zenodo.10254871)

# Prosper

We present the package prosper\_nn, that provides four neural network architectures dedicated to time series forecasting, implemented in PyTorch: the Error Correction Neural Network (ECNN), the Historical Consistent Neural Network (HCNN), the Causal-Retro-Causal Neural Network (CRCNN), and the Fuzzy Neural Network. In addition, prosper\_nn contains the first sensitivity analysis suitable for RNNs and a heatmap to visualize forecasting uncertainty which was previously only available in Java.
These models and methods are used in industry for two decades and were used and referenced in several scientific publications. However, only now we make them publicly available, allowing researchers and practitioners to benchmark and further develop them.	The package is designed to make the models easily accessible, thereby enabling research and application.
The full documentation can be found on https://fraunhofer-iis.github.io/prosper_nn/. There are also tutorials that show how to work with the package.
## Installation

You can use pip to install the package:

`pip install prosper-nn`

Afterwards you can import prosper_nn with:

`import prosper_nn`

## Contribution

Everyone is invited to contribute to the package. If you want to do so, prior to creating a pull request, please download the [contributor license agreement](docs/source/contriubte/prosper_nn_cla.pdf).
After you signed the file, please send it to prosper_nn@iis.fraunhofer.de. Afterwards, the maintainer will consider your pull request.

## Acknowledgments

This work was supported by the Bavarian Ministry of Economic Affairs, Regional Development and Energy through the Center for Analytics – Data – Applications (ADA-Center) within the framework of „BAYERN DIGITAL II“ (20-3410-2-9-8)
