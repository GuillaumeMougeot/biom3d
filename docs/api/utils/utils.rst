Utils
=====

The utils module define functions (or class) that can be used in several other modules, needed by the core of biom3d or not numerous enough to constitute a new module.

.. note:: 

      Altough we present submoduls, you don't need to specify them when importing :

      .. code-block:: python

            from biom3d.utils.data_handler_abstract import DataHandler 
            # Is the same thing as
            from biom3d.utils import DataHandler

      Only some functions/classes cannot be imported directly with utils.

.. toctree::
      :maxdepth: 2
      :caption: Utils 

      datahandler.rst
      config.rst
      sampling.rst
      image.rst
      filtering.rst
      encoding.rst
