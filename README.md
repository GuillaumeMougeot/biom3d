<div align="center">
  <img src="https://github.com/GuillaumeMougeot/biom3d/blob/main/images/logo_biom3d_crop.png" width="200" title="biom3d" alt="biom3d" vspace = "0">

  [📘Documentation](https://biom3d.readthedocs.io/) | 
  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/GuillaumeMougeot/biom3d/blob/master/docs/biom3d_colab.ipynb)
</div>
<img src="https://github.com/GuillaumeMougeot/biom3d/blob/main/images/nucleus_segmentation.png" width="200" title="nucleus" alt="nucleus" align="right" vspace = "0">

<!-- [**Documentation**](https://biom3d.readthedocs.io/) -->

<!-- **Try it online!** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/GuillaumeMougeot/biom3d/blob/master/docs/biom3d_colab.ipynb) -->

## Highlights

Biom3d automatically configures the training of a 3D U-Net for 3D semantic segmentation.

The default configuration matches the performance of [nnUNet](https://github.com/MIC-DKFZ/nnUNet) but is much easier to use both for community users and developers. Biom3d is flexible for developers: easy to understand and easy to edit. 

Code architecture of Biom3d versus code architecture of nnU-Net:

Biom3d modules             |  nnUNet modules
:-------------------------:|:-------------------------:
![](https://github.com/GuillaumeMougeot/biom3d/blob/main/images/biom3d_train.png)  |  ![](https://github.com/GuillaumeMougeot/biom3d/blob/main/images/nnunet_run_run_training.png)

*Illustrations generated with `pydeps` module*

> **Disclaimer**: Biom3d does not include the possibility to use 2D U-Net or 3D-Cascade U-Net or Pytorch distributed parallel computing (only Pytorch Data Parallel) yet. However, these options could easily be adapted if needed.

We target two main types of users:

* Community users, who are interested in using the basic functionalities of Biom3d: GUI or CLI, predictions with ready-to-use models or default training.
* Deep-learning developers, who are interested in more advanced features: changing default configuration, writing of new Biom3d modules, Biom3d core editing etc.

**[21/11/2023] NEWS!** Biom3d tutorials are now available online:

* [I2K Workshop tutorial (in english)](https://www.youtube.com/watch?v=cRUb9g66P18&ab_channel=I2KConference)
* [RTMFM tutorial (in french)](https://www.youtube.com/live/fJopxW5vOhc?si=qdpJcaEy0Bd2GDec)

## 🔨 Installation

**For the installation details, please check our documentation here:** [**Installation**](https://biom3d.readthedocs.io/en/latest/installation.html)

TL;DR: here is a single line of code to install biom3d:

```
pip install torch biom3d
```

## ✋ Usage

**For Graphical User Interface users, please check our documentation here:** [**GUI**](https://biom3d.readthedocs.io/en/latest/quick_run_gui.html)

**For Command Line Interface users, please check our documentation here:** [**CLI**](https://biom3d.readthedocs.io/en/latest/tuto_cli.html)

**For Deep Learning developers, the tutorials are currently being cooked stayed tuned! You can check the partial API documentation already:** [**API**](https://biom3d.readthedocs.io/en/latest/builder.html)

TL;DR: here is a single line of code to run biom3d on the [BTCV challenge](https://www.synapse.org/#!Synapse:syn3193805/wiki/217785) and reach the same performance as nnU-Net (no cross-validation yet): 

```
python -m biom3d.preprocess_train\
 --img_dir data/btcv/Training/img\
 --msk_dir data/btcv/Training/label\
 --num_classes 13\
 --ct_norm
```

## ⚠ Disclaimer

> **Warning**: This repository is still a work in progress and comes with no guarantees.

## Issues

Please feel free to open an issue or send me an email if any problem with biom3d appears. But please make sure first that this problem is not referenced on the FAQ page: [Frequently Asked Question](https://biom3d.readthedocs.io/en/latest/faq.html)

## 📑 Citation

If you find Biom3d useful in your research, please cite:

```
@misc{biom3d,
  title={{Biom3d} Easy-to-use Tool for 3D Semantic Segmentation of Volumetric Images using Deep Learning},
  author={Guillaume Mougeot},
  howpublished = {\url{https://github.com/GuillaumeMougeot/biom3d}},
  year={2023}
  }
```

## 💰 Fundings and Acknowledgements 

This project has been inspired by the following publication: "nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation", Fabian Isensee et al, Nature Method, 2021.

This project has been supported by Oxford Brookes University and the European Regional Development Fund (FEDER). It was carried out between the laboratories of iGReD (France), Institut Pascal (France) and Plant Nuclear Envelop (UK).

<p align="middle">
  <img src="https://github.com/GuillaumeMougeot/biom3d/blob/main/images/Flag_of_Europe.svg.png" alt="Europe" width="100">
  <img src="https://github.com/GuillaumeMougeot/biom3d/blob/main/images/brookes_logo_black.bmp" alt="Brookes" width="100">
  <img src="https://github.com/GuillaumeMougeot/biom3d/blob/main/images/GReD_color_EN.png" alt="iGReD" width="100">
  <img src="https://github.com/GuillaumeMougeot/biom3d/blob/main/images/logo_ip.png" alt="IP" width="100">
  <img src="https://github.com/GuillaumeMougeot/biom3d/blob/main/images/logo_aura.PNG" alt="AURA" width="100">
  <img src="https://github.com/GuillaumeMougeot/biom3d/blob/main/images/logo_UCA.jpg" alt="UCA" width="100">
</p>



