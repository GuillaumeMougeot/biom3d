Biom3d
===================================

.. image:: _static/image/nucleus_segmentation.png
    :align: right
    :height: 200

Congrats! You've just found Biom3d, an easy-to-use tool for volumetric image semantic segmentation.

This tool is addressed to three different profiles:

- Non Programmers, who could be interested to use the Graphical User Interface (GUI) only. After going through the installation [link], start with [Quick Run/GUI].
- Python Programmers, who could be happy with some minimal customization, such as adapting Biom3d to a novel image format or choosing another training loss. After going through the installation [link], start with [CLI].
- Deep Learning Programmers, who are not scared of digging in some more advanced features of Biom3d, such as customizing the deep learning model or the metrics. After going through the installation [link], start with the basic here [CLI] and then go directly to the tutorials [link].

.. toctree::
   :maxdepth: 2
   :caption: Installation

   installation.md

.. toctree::
   :maxdepth: 2
   :caption: Tutorials

   quick_run_gui.md
   tuto_cli.md

.. toctree::
   :maxdepth: 2
   :caption: FAQ

   faq.md

.. toctree::
   :maxdepth: 2
   :caption: API 

   preprocess.rst
   auto_config.rst
   builder.rst
   metrics.rst
   callbacks.rst
   trainers.rst
   predictors.rst
