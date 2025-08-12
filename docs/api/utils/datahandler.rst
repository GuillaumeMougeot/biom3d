DataHandler
===========

DataHandler is the class to use to load, save and iterate on images, masks or foreground.

.. currentmodule:: biom3d.utils

Instanciating a DathaHandler with the :class:`DataHandlerFactory`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you want to use a :class:`DataHandler`, we strongly recommend using our factory. 

.. autoclass:: DataHandlerFactory
    :members:

.. currentmodule:: biom3d.utils.data_handler.data_handler_abstract

The :class:`DataHandler` contract
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Here we will describe the abstract class :class:`DataHandler`, so you can use it or implement a new one.

Publics attributes
------------------
.. autoattribute:: DataHandler.images
.. autoattribute:: DataHandler.masks
.. autoattribute:: DataHandler.fg
.. autoattribute:: DataHandler.msk_outpath

Privates attributes
-------------------
.. autoattribute:: DataHandler._images_path_root
.. autoattribute:: DataHandler._masks_path_root
.. autoattribute:: DataHandler._fg_path_root
.. autoattribute:: DataHandler._image_index
.. autoattribute:: DataHandler._iterator
.. autoattribute:: DataHandler._size
.. autoattribute:: DataHandler._saver

Public methods
--------------
.. automethod:: DataHandler.open
.. automethod:: DataHandler.close
.. automethod:: DataHandler.get_output
.. automethod:: DataHandler.load
.. automethod:: DataHandler.save
.. automethod:: DataHandler.insert_prefix_to_name
.. automethod:: DataHandler.reset_iterator


Private methods
---------------
.. automethod:: DataHandler._input_parse
.. automethod:: DataHandler._output_parse
.. automethod:: DataHandler._output_parse_preprocess
.. automethod:: DataHandler._save

Specials methods
----------------
.. automethod:: DataHandler.__init__
.. automethod:: DataHandler.__iter__
.. automethod:: DataHandler.__next__
.. automethod:: DataHandler.__len__
.. automethod:: DataHandler.__del__

:class:`OutputType`
~~~~~~~~~~~~~~~~~~~

.. autoclass:: OutputType
.. autoattribute:: OutputType.IMG
.. autoattribute:: OutputType.MSK
.. autoattribute:: OutputType.FG
.. autoattribute:: OutputType.PRED

Adding a new dataset format
~~~~~~~~~~~~~~~~~~~~~~~~~~~

To add a new format, only two thing are required :

- Create a new implementation of :class:`DataHandler`.

  .. note::

      Redefine all abstract methods, if you think one of the other methods need a redefinition, do it, just respect the contract.
      You can use existing implementations as base.

- Add some code to the :class:`DataHandlerFactory` to allow it to recognize your new implementation. 
- Document your format in `docs/tuto/dataset.md`, specially if your implementation need a specific dataset structure.

.. note::

    When testing, be sure to also test with dataset of only 1 image to test if preprocessing._split_image work well.

Adding a new image format
~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. currentmodule:: biom3d.utils.data_handler.file_handler

In case you work on file using :class:`FileHandler` and you need to use another format than Numpy, TIFF or Nifty, you can easily implement it.

In the module `biom3d.utils.data_handler.file_handler`, there is a static class named :class:`ImageManager`. This class implement the methods to read and save a single image as a file.

Two functions will interest us :

.. autoclass:: ImageManager
    :members: adaptive_imread, adaptive_imsave

To implement a new file format for image (for example png because why not) you simply have to add the possibility for those two function to treat the new format, then it is all automatic.

.. note:: 

    We strongly advise to create two separate private function, one for reading and another one for saving, and call them in adaptive.