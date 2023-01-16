Create Data for Recurrent Networks
-----------------------------------

Module
^^^^^^

.. automodule:: prosper_nn.utils.generate_time_series_data
    :members:
    :undoc-members:
    :show-inheritance:

Example
^^^^^^^

.. code:: python

    >>> Y, U = sample_data(100, 1, 3)
    >>> print(Y.shape)
    >>> print(U.shape)
    torch.Size([100, 1])
    torch.Size([100, 3])
    