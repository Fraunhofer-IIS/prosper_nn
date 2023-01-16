Historical Consistent Neural Network
------------------------------------

Module
^^^^^^

.. automodule:: prosper_nn.models.hcnn.hcnn
    :members:
    :undoc-members:
    :show-inheritance:

Example
^^^^^^^

.. literalinclude:: ../examples_for_doc/hcnn_example.py
    :language: python

.. code:: python

    hcnn = models.hcnn.HCNN(5, 1, 20, 5, 1)
    input = torch.randn(20, 1, 1)
    past_error, forecast = torch.split(hcnn(input), 20)

Reference
^^^^^^^^^

`Zimmermann HG., Tietz C., Grothmann R. (2012) Forecasting with Recurrent Neural Networks: 12 Tricks.
In: Montavon G., Orr G.B., Müller KR. (eds) Neural Networks: Tricks of the Trade.
Lecture Notes in Computer Science, vol 7700. Springer, Berlin, Heidelberg.
https://doi.org/10.1007/978-3-642-35289-8_37 <https://link.springer.com/chapter/10.1007/978-3-642-35289-8_37>`_

.. _Historical Consistent Neural Network Cell:

Historical Consistent Neural Network Cell
-----------------------------------------

Module
^^^^^^

.. automodule:: prosper_nn.models.hcnn.hcnn_cell
    :members:
    :undoc-members:
    :show-inheritance:

Example
^^^^^^^

.. code:: python

    hcnn_cell = model.hcnn.HCNNCell(5, 1)
    observation = torch.randn(1, 1)
    state = torch.randn(1, 5)
    outputs = []
    for i in range(6):
        state, output = hcnn_cell(state, observation)
        outputs.append(output)

.. _Historical Consistent Neural Network LSTM Cell:

Historical Consistent Neural Network LSTM Cell
----------------------------------------------

Module
^^^^^^

.. automodule:: prosper_nn.models.hcnn.hcnn_lstm_cell
    :members:
    :undoc-members:
    :show-inheritance:

Example
^^^^^^^

.. code:: python

    hcnn_cell = model.hcnn.HCNN_LSTM_Cell(10, 20)
    observation = torch.randn(1, 1)
    state = torch.randn(1, 5)
    outputs = []
    for i in range(6):
        state, output = hcnn_cell(state, observation)
        outputs.append(output)