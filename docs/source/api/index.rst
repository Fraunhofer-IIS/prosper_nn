API Reference
=============

In this section, implemented classes are explained and their methods are described. In addition, the utils functions implemented in this package are described.
For these classes and functions, the necessary and optional parameters are documented here.

The package consists of two subpackages with different purposes. The models folder contains different architectures of neural networks implemented in PyTorch, including different neural networks classes for predicting regression tasks or time series forecasting.
It also includes a wrapper to build ensembles of models, where the outputs of multiple models are averaged.

The second subpackage contains utils functions. There are functions for data creation and formatting, but also for evaluating and investigating your trained model, including an analysis of the hidden-layer correlation, input/output sensitivity and visualization.


Models
------

Implemented models for regression tasks are:

.. toctree::
    :maxdepth: 1

    feedforward
    deepff

If you have only a small number of training data or need explainability of the model, have a look at:

.. toctree::
    :maxdepth: 1

    fuzzy

There are multiple neural networks (and their according cells) implemented for time series forecasting:

.. toctree::
    :maxdepth: 2

    ecnn
    hcnn
    dhcnn
    crcnn
    hcnn_known_u

The Ensemble is applicable for all the above models and a lot more:

.. toctree::
    :maxdepth: 1

    ensemble

Utils
-----

The utils subpackage contains functions to generate and prepare data for training:

.. toctree::
    :maxdepth: 1

    create_input_ecnn_hcnn
    generate_time_series_data

The following utils are interesting for evaluating and investigating your neural network or visualize forecasts with or without uncertainty:

.. toctree::
    :maxdepth: 1

    neuron_correlation_hidden_layer
    sensitivity_analysis
    visualize_forecasts
    visualization
