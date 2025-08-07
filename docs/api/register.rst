Register
========

The register is a core module of Biom3d as it define what function can be used for every module. 

.. note:: 
    
    If you implement a new function/module, you may want to integrate it directly in Biom3d by adding it to the register. If so, you can add it's documentation here and submit a pull request.

.. literalinclude:: ../../src/biom3d/register.py
   :language: python
   :linenos:

Now let's delve on each modules.

Dataloader and batchgenerator
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

There are currently 2 dataloader and 1 batchgenerator.

.. currentmodule:: biom3d.datasets.semseg_patch_fast

SegPatchFast
------------

This is the default dataloading module.

.. autoclass:: SemSeg3DPatchFast 
    :no-index:

.. currentmodule:: biom3d.datasets.semseg_torchio

Torchio
-------

This dataloader is an implementation of torchio subjectloader, optimized to do operation with torchio. It was written to ease data augmentation but is not used.

.. autoclass:: TorchioDataset 
    :no-index:

.. currentmodule:: biom3d.datasets.semseg_batchgen

BatchGen
--------

This is nnUnet batchgenerator. It has been slightly modified to act like a dataloader to be easier to interchange with the others.

.. note::
   
   If you want to use it, you have to modify the config file in a different way:
   
   .. code-block:: python

       TRAIN_DATALOADER = Dict(
           fct="BatchGen",
           kwargs=Dict(
               # Insert parameters here
           )
       )
       VAL_DATALOADER = Dict(
           fct="BatchGen",
           kwargs=Dict(
               # Insert parameters here
           )
       )
   
        # Instead of 

        TRAIN_DATASET = Dict(
            fct="SegPatchFast",
            kwargs=Dict(
                # Insert parameters here
            )
        )
        VAL_DATASET = Dict(
            fct="SegPatchFast",
            kwargs=Dict(
                # Insert parameters here
            )
        )


.. autoclass:: MTBatchGenDataLoader 
    :no-index:

.. note:: 
    This class use a dependency that is not in Biom3d's dependency so you will need to install it manually : `pip install batchgenerators`

Models
~~~~~~

.. currentmodule:: biom3d.models.unet3d_vgg_deep

.. class:: UNet

---------------

This is a transcription of nnUnet neural network, and is the default model used by Biom3d

.. autoclass:: UNet 
    :no-index:

.. currentmodule:: biom3d.models.encoder_vgg

It use :class:`VGGEncoder` as encoder :

.. autoclass:: VGGEncoder 
    :no-index:

And it use :class:`VGGDecoder` as decoder 

.. currentmodule:: biom3d.models.unet3d_eff

.. class:: EfficientNet3D

---------------

This is a transcription of nnUnet neural network, and is the default model used by Biom3d

.. autoclass:: EfficientNet3D 
    :no-index:

.. currentmodule:: biom3d.models.encoder_efficientnet3d

It use :class:`EfficientNet3D` as encoder :

.. autoclass:: EfficientNet3D 
    :no-index:

.. currentmodule:: biom3d.models.decoder_vgg_deep

And it use :class:`VGGDecoder` as decoder.

.. note:: 
    
    Both the encoder can also be used as models.

Metrics
~~~~~~~
Here are the possibles metrics :

.. currentmodule:: biom3d.metrics

Dice
----

One of the most used metrics. It is a comparison between the two image returning a number between `0` and `1`, the closer it is to `1`, the closer are the images.

.. autoclass:: Dice 
    :no-index:

CrossEntropy
------------

Metric that compare the softmax of the logit and the mask.

.. autoclass:: CrossEntropy 
    :no-index:

DiceBCE
-------

Metric that ally :class:`Dice` and :class:`CrossEntropy`

.. autoclass:: DiceBCE 
    :no-index:

DC_and_CE_loss
--------------

nnUnet's implementation of :class:`Dice` and :class:`CrossEntropy`, is more robust but doesn't treat binary cases.

.. autoclass:: DC_and_CE_loss 
    :no-index:

IoU
---

One of the most used metrics. It is a comparison between the two image returning a number between `0` and `1`, the closer it is to `1`, the closer are the images.
It is close the :class:`Dice` but with less weight on the intersection (eg: Dice of 0.5 while IoU of 0.66).

.. autoclass:: IoU 
    :no-index:

MSE
---

Use mean square method to compute loss.

.. autoclass:: MSE 
    :no-index:

DeepMetric
----------

A deep supervision metric. Can be used with :class:`DiceBCE` and :class:`MSE`.

.. autoclass:: DeepMetric 
    :no-index:

Trainers
~~~~~~~~

Here are the possibles trainers :

.. currentmodule:: biom3d.trainers

SegTrain
--------

Default trainer, do the whole training.

.. autofunction:: seg_train 
    :no-index:

SegVal
------

Default validater.

.. autofunction:: seg_validate 
    :no-index:

SegPatchTrain
-------------

Torchio trainer, created to use Torchio datasets and patch approach.

.. autofunction:: seg_patch_train 
    :no-index:

SegPatchVal
-----------

Torchio validater, created to use Torchio datasets and patch approach.

.. autofunction:: seg_patch_validate
    :no-index:

Preprocessor
~~~~~~~~~~~~

There is currently only one preprocessor :

.. currentmodule:: biom3d.preprocess

.. autofunction:: seg_preprocessor
    :no-index:

Predictors
~~~~~~~~~~

Here are the available predictors :

There is currently only one preprocessor :

.. currentmodule:: biom3d.predictors

Seg
---

.. note:: 
    This predictor should not be used, it is only here for retrocompatibility sake.

.. autofunction:: seg_predict
    :no-index:

SegPatch
--------

.. note:: 
    This is the default predictor.

.. autofunction:: seg_predict_patch_2
    :no-index:

Postprocessor
~~~~~~~~~~~~~

There is currently only one postprocessor :

.. currentmodule:: biom3d.predictors

.. autofunction:: seg_postprocessing
    :no-index: