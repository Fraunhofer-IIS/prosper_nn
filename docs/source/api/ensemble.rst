Ensemble
--------

Module
^^^^^^

.. automodule:: prosper_nn.models.ensemble.ensemble
    :members:
    :undoc-members:
    :show-inheritance:


Example
^^^^^^^

.. code-block:: python

    # replace simple model with your architecture
    ensemble = Ensemble(simple_model, n_models)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(ensemble.parameters())
    for t in range(epochs):
        # Forward pass: Compute predicted y by passing x to the ensemble
        y_preds, mean_y = torch.split(ensemble(x), n_models)

        optimizer.zero_grad()
        # compute loss for each model instance
        loss = sum([criterion(y_pred, y) for y_pred in y_preds])
        loss.backward()
        optimizer.step()