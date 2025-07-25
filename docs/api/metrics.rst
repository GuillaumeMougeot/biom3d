Metrics
=======

The metrics are used to both train the model, they will be called 'loss', and monitor the training process. All metrics must inherit from the `biom3d.metrics.Metric` class. Once defined a novel metric can be either used as a loss function and integrated in the config file as follows:

>>> TRAIN_LOSS = Dict(
>>>     fct="DiceBCE",
>>>     kwargs = Dict(name="train_loss", use_softmax=True)
>>> )

or as a metric with the following:

>>> VAL_METRICS = Dict(
>>>     val_iou=Dict(
>>>         fct="IoU",
>>>         kwargs = Dict(name="val_iou", use_softmax=USE_SOFTMAX)),
>>>     val_dice=Dict(
>>>         fct="Dice",
>>>         kwargs=Dict(name="val_dice", use_softmax=USE_SOFTMAX)),
>>> )

.. automodule:: biom3d.metrics
    :members: