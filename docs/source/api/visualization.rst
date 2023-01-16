Visualization of Heatmaps
-------------------------

Module
^^^^^^

.. automodule:: prosper_nn.utils.visualization
    :members:
    :undoc-members:
    :show-inheritance:

Example
^^^^^^^ 

.. code:: python

    X = torch.rand([10, 100])
    plot_heatmap(X,
                 xlabel={'xlabel': 'xlabel'},
                 ylabel={'ylabel': 'ylable'},
                 title={'label': 'Title'})