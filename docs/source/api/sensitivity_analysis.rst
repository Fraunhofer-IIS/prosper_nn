Sensitivity Analysis
--------------------

Modules
^^^^^^^

.. automodule:: prosper_nn.utils.sensitivity_analysis
    :members:
    :undoc-members:
    :show-inheritance:

Example
^^^^^^^

.. code:: python

    X = torch.rand([10, 100]) # shape=(n_observations, input_dim)
    net = prosper_nn.models.feedforward.FFNN(100, 5, 1)

    # with trained model
    sensitivity_analysis(net, X)