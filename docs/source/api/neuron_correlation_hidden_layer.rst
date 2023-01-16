Neuron Correlation Hidden Layer
-------------------------------

Modules
^^^^^^^

.. automodule:: prosper_nn.utils.neuron_correlation_hidden_layers
    :members:
    :undoc-members:
    :show-inheritance:

Example
^^^^^^^

.. code:: python

    X = torch.rand([10, 100])
    net = prosper_nn.models.feedforward.FFNN(100, 5, 1)

    # with trained model
    net.eval()
    corr_matrix2, max_corr2, ind_neurons2 = hl_size_analysis_Sequential(
        net, name_layer="hidden", model_input=X)
