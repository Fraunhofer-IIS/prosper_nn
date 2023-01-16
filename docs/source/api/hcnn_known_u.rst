Historical Consistent Neural Network With_Know_U
-------------------------------------------------

Module
^^^^^^

.. automodule:: prosper_nn.models.hcnn_known_u.hcnn_known_u
    :members:
    :undoc-members:
    :show-inheritance:

Example
^^^^^^^

.. literalinclude:: ../examples_for_doc/hcnn_known_u_example.py
    :language: python

.. code:: python

    hcnn_known_u_model = hcnn_known_u.HCNN_KNOW_U(50, 3, 5, 10, 5)
    input = torch.randn(20, 1, 1)
    past_error, forecast = torch.split(hcnn(input), 20)
