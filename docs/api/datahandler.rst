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

Adding a new format
~~~~~~~~~~~~~~~~~~~

To add a new format, only two thing are required :
- Create a new implementation of :class:`DataHandler`.

  .. note::

      Redefine all abstract methods, if you think one of the other methods need a redefinition, do it, just respect the contract.
      You can use existing implementations as base.

- Add some code to the :class:`DataHandlerFactory` to allow it to recognize your new implementation. 