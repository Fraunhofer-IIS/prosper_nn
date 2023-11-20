.. Prosper documentation master file, created by
   sphinx-quickstart on Tue Sep 22 14:39:02 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.


###########
Prosper\_nn
###########

Problem-Specific Pre-Structuring of Neural Networks
###################################################

We present the package prosper_nn, that provides four neural network architectures dedicated to time series forecasting, implemented in PyTorch: the Error Correction Neural Network (ECNN), the Historical Consistent Neural Network (HCNN), the Causal-Retro-Causal Neural Network (CRCNN), and the Fuzzy Neural Network. In addition, prosper_nn contains the first sensitivity analysis suitable for RNNs and a heatmap to visualize forecasting uncertainty which was previously only available in Java. These models and methods are used in industry for two decades and were used and referenced in several scientific publications. However, only now we make them publicly available, allowing researchers and practitioners to benchmark and further develop them. The package is designed to make the models easily accessible, thereby enabling research and application. The Github repository is hosted on `https://github.com/Fraunhofer-IIS/prosper_nn <https://github.com/Fraunhofer-IIS/prosper_nn>`_.

This is the documentation site of prosper_nn. It is shown how to `install the package <install/index.rst>`_.
Also we provide an in detail description of the models and functions in `API References <api/index.rst>`_.
Furthermore, you can have a look at different `tutorials <tutorials/index.rst>`_ that explain the theory of the models and show how to work with the package to build powerful neural networks.


.. toctree::
   :maxdepth: 2
   :caption: Contents

   install/index
   api/index
   tutorials/index
   contribute/index
   authors
   license


******************
Acknowledgments
******************

This work was supported by the Bavarian Ministry of Economic Affairs, Regional Development and Energy through the Center for Analytics – Data – Applications (ADA-Center) within the framework of „BAYERN DIGITAL II“ (20-3410-2-9-8)

******************
Indices and tables
******************

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
