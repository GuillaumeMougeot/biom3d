Trainers
========

The Trainers are Python functions that take as input a dataloader, a model, a loss function and an optimizer function to start a training process. Optionally, a list of `biom3d.metrics.Metric <https://biom3d.readthedocs.io/en/latest/metrics.html>`_ and a `biom3d.callback.Callbacks <https://biom3d.readthedocs.io/en/latest/callbacks.html>`_ can be provided to the trainer to enrich the training loop. 

Validaters, which optionally perform validation in the end of each epoch, are also defined in `biom3d.trainers`.

.. automodule:: biom3d.trainers
    :members:
