.. Prosper documentation master file, created by
   sphinx-quickstart on Tue Sep 22 14:39:02 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.


###########
Prosper\_nn
###########

Problem-Specific Pre-Structuring of Neural Networks
###################################################

Accurate data-driven forecasts can provide a crucial advantage in many application areas.
One of the methods with the most promising results in forecasting time series are neural networks.
However, especially in macro-economic applications,
it can be difficult and time-consuming to adapt state-of-the-art neural network architectures in a way that leads to satisfying results.
For instance, the final prices of materials and stocks result from a highly complex interplay between supply and demand.
Additionally, there is often only one (albeit long) historical time series available for training, which makes correlations in the data difficult to detect.

Under these circumstances, applying state-of-the-art neural networks architectures successfully poses a great challenge.
Pre-structuring the models can solve this problem. For this purpose, Zimmermann, Tietz and Grothmann (Neural Networks: Tricks of the Trade, 2012)
propose recurrent architectures for various time series problems that help recognize correlations.
They recommend Error-Correction Neural Networks (ECNNs), Historical-Consistent Neural Networks (HCNNs) and Causal-Retro-Causal Neural Networks (CRCNNs).
One of the main ideas of the pre-structuring is embedding the model in a larger architecture in order to use the past prediction errors for predicting the next time step. The three approaches mentioned use this idea and apply it in different settings.
So far, the proposed architectures are not publicly available in common machine learning frameworks.
Therefore, we have implemented the models in PyTorch. This way, we can easily test them on diverse datasets.

In addition to these models for time series forecasting, we have also implemented a fuzzy neural network. This is also a pre-structured model, but this time with the goal of interpretability.

This is the documentation site of this package. It is shown how to `install the package <install/index.rst>`_.
Also we provide an in detail description of the the models and functions in `API References <api/index.rst>`_.
Furthermore, you can have a look at different `tutorials <tutorials/index.rst>`_ that explain the theory of the models and show how to work with the package to build powerful neural networks.


.. toctree::
   :maxdepth: 2
   :caption: Contents

   install/index
   api/index
   tutorials/index
   authors
   license

******************
Indices and tables
******************

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
