Error Correction Neural Network
-------------------------------

Module
^^^^^^

.. automodule:: prosper_nn.models.ecnn.ecnn_model
    :members:
    :undoc-members:
    :show-inheritance:

Example
^^^^^^^

.. literalinclude:: ../examples_for_doc/ecnn_example.py
    :language: python

Reference
^^^^^^^^^

`Zimmermann HG., Tietz C., Grothmann R. (2012) Forecasting with Recurrent Neural Networks: 12 Tricks.
In: Montavon G., Orr G.B., Müller KR. (eds) Neural Networks: Tricks of the Trade.
Lecture Notes in Computer Science, vol 7700. Springer, Berlin, Heidelberg.
https://doi.org/10.1007/978-3-642-35289-8_37 <https://link.springer.com/chapter/10.1007/978-3-642-35289-8_37>`_

Error Correction Neural Network Cell
------------------------------------

Module
^^^^^^

.. automodule:: prosper_nn.models.ecnn.ecnn_cell
    :members:
    :undoc-members:
    :show-inheritance:

Example
^^^^^^^

.. code:: python

    ecnn_cell = model.ecnn.ECNNCell(5, 10)
    state = torch.randn(1, 5)
    U = torch.randn(1, 5)
    Y = torch.randn(1, 1)
    outputs = []
    for i in range(6):
        state, output = ecnn_cell(state, U, Y)
        outputs.append(output)

Reference
^^^^^^^^^

`Zimmermann HG., Tietz C., Grothmann R. (2012) Forecasting with Recurrent Neural Networks: 12 Tricks.
In: Montavon G., Orr G.B., Müller KR. (eds) Neural Networks: Tricks of the Trade.
Lecture Notes in Computer Science, vol 7700. Springer, Berlin, Heidelberg.
https://doi.org/10.1007/978-3-642-35289-8_37 <https://link.springer.com/chapter/10.1007/978-3-642-35289-8_37>`_

Error Correction Neural Network LSTM Cell
-----------------------------------------

Module
^^^^^^

.. automodule:: prosper_nn.models.ecnn.ecnn_lstm_cell
    :members:
    :undoc-members:
    :show-inheritance:

Example
^^^^^^^

.. code:: python

    ecnn_cell = model.ecnn.ECNN_LSTM_Cell(5, 10)
    state = torch.randn(1, 5)
    U = torch.randn(1, 5)
    Y = torch.randn(1, 1)
    outputs = []
    for i in range(6):
        state, output = ecnn_cell(state, U, Y)
        outputs.append(output)

Reference
^^^^^^^^^

`Zimmermann HG., Tietz C., Grothmann R. (2012) Forecasting with Recurrent Neural Networks: 12 Tricks.
In: Montavon G., Orr G.B., Müller KR. (eds) Neural Networks: Tricks of the Trade.
Lecture Notes in Computer Science, vol 7700. Springer, Berlin, Heidelberg.
https://doi.org/10.1007/978-3-642-35289-8_37 <https://link.springer.com/chapter/10.1007/978-3-642-35289-8_37>`_