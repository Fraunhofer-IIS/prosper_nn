Tutorials
=========

These notebooks give you an overview about how to use the different models implemented in the Prosper\_nn package.
The whole workflow is described. This means they can be used as a guide to build powerful neural networks.

We present notebooks for different types of tasks. The basic tutorial is about `Regression Neural Networks <Regression Flow.ipynb#Regression-Neural-Network-Workflow>`_.
It shows how a regression task can be solved with Feedforward Neural Networks.
The concepts are explained in small detail, which makes it an ideal starting point to get to know the package.
The other tutorials are sometimes based on or refer to it.

If you have only a small number of available data and need explainability of the model, have a look at (Fuzzy Neural Networks).

For time series forecasting there are multiple options. The Error Correction Neural Network tutorial is the right one if you want to forecast only a small number of time steps into the future.
If you wish to forecast for a long horizon, you should probably look into the `Historical Consistant Neural Network <HCNN.ipynb#Historical-Consistent-Neural-Network-Workflow>`_ tutorial.
If the results aren't good enough, giving the `Causal-Retro Causal <CRCNN.ipynb#Causal-Retro-Causal-Neural-Network-Workflow>`_ model a shot might be a good idea.


The notebooks begin with simple versions of the models and present further developments of them later in the tutorials.
For each model it starts with the theory behind them and the basic concepts are explained.
This is followed by the data preparation, which shows the way the data should look like so the models can work with it.
Then, for each model a training loop is presented which demonstrates how to train the neural network.
Afterwards, the model is evaluated. Here, we try to get insights into the model or visualize the predictions or forecasts.


.. toctree::
    :maxdepth: 2

    Regression
    ECNN
    Hcnn
    CRCNN
    Neuro-Fuzzy
    Hcnn_known_u
