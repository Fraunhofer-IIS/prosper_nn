Create Input for Recurrent Networks
-----------------------------------

Module
^^^^^^

.. automodule:: prosper_nn.utils.create_input_ecnn_hcnn
    :members:
    :undoc-members:
    :show-inheritance:

Example
^^^^^^^

.. code:: python

    >>> Y, U = sample_data(100, 1, 3)
    >>> Y_batches, U_batches = ci.create_input(Y, 20, 2, U, False)
    >>> print(Y_batches.shape)
    >>> print(U_batches.shape)
    torch.Size([5, 20, 2, 1])
    torch.Size([5, 20, 2, 3])