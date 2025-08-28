"""
3D efficient net stolen from: https://github.com/shijianjian/EfficientNet-PyTorch-3D.

Usage:

.. code-block:: python

    model = EfficientNet3D.from_name("efficientnet-b1", override_params={'include_top': False}, in_channels=1)
    model.cuda() # On CUDA machine
    model.to('mps') # On Apple Silicon

"""

from typing import Callable, Optional
from biom3d.models.encoder_vgg import EncoderBlock
from biom3d.models.decoder_vgg_deep import VGGDecoder
from biom3d.models.encoder_efficientnet3d import EfficientNet3D, efficientnet3d

import torch
from torch import nn

def get_layer(model:nn.Module, layer_names:list[str])->nn.Module:
    """
    Retrieve a submodule from a model based on a list of keys.

    Parameters
    ----------
    model : nn.Module
        The PyTorch model to search within.
    layer_names : list of str
        List of submodule names to traverse, e.g. ['_blocks', '0', '_depthwise_conv'].

    Returns
    -------
    nn.Module
        The requested submodule.
    """
    for e in layer_names: 
        model = model._modules[e]
    return model

def get_pyramid(model:nn.Module, pyramid:dict)->list[nn.Module]:
    """
    Retrieves multiple submodules from a model according to a dictionary of paths.

    Parameters
    ----------
    model : nn.Module
        The model to extract layers from.
    pyramid : dict
        Dictionary where each value is a list of strings indicating a submodule path.

    Examples
    --------
    >>> pyramid = {
    ...     0: ['_conv_stem'],              # 100
    ...     1: ['_blocks', '1', '_bn0'],    # 50
    ...     2: ['_blocks', '3', '_bn0'],    # 25
    ...     3: ['_blocks', '5', '_bn0'],    # 12
    ...     4: ['_blocks', '11', '_bn0'],   # 6
    ...     5: ['_bn1']                     # 3
    ... }

    Returns
    -------
    list of nn.Module
        List of layers (submodules) extracted from the model.
    """
    layers = []
    for v in pyramid.values():
        layers += [get_layer(model, v)]
    return layers

def get_outfmaps(layer:nn.Module)->int:
    """
    Returns the depth of output feature maps of a layer.

    Parameters
    ----------
    layer : nn.Module
        The layer to inspect.

    Returns
    -------
    int
        Number of output feature maps (channels).

    Notes
    -----
    Tries to read from 'num_features' or 'in_channels' attributes. Returns 0 on failure.
    """
    if 'num_features' in layer.__dict__.keys():
        return layer.num_features
    elif 'in_channels' in layer.__dict__.keys():
        return layer.in_channels
    else:
        print("[Error] layer is not standard, cannot extract output feature maps.")
        return 0

#---------------------------------------------------------------------------
# 3D UNet with the previous encoder and decoder

class EffUNet(nn.Module):
    """
    3D U-Net model using EfficientNet3D as encoder and VGG-style decoder.

    This model builds a pyramid of intermediate feature maps from the encoder, and
    passes them to the decoder for semantic segmentation.

    :ivar EfficientNet3D encoder: EfficientNet3D encoder model.
    :ivar list pyramid: List of intermediate encoder layers used for skip connections.
    :ivar dict down: Dictionary mapping pyramid levels to encoder activations (populated via forward hooks).
    :ivar torch.nn.Module decoder: VGG-style decoder module.
    """

    def __init__(
        self, 
        patch_size:int|tuple[int], # TODO: Clement: Guillaume this should be a tuple (or something like it) but the whole code of the encoder is considering it as an int, we need to make it clear
        num_pools:list[int]=[5,5,5], 
        num_classes:int=1, 
        factor:int=32,
        encoder_ckpt:Optional[str] = None,
        model_ckpt:Optional[str] = None,
        use_deep:bool=True,
        in_planes:int = 1,
        ):
        """
        3D U-Net model using EfficientNet3D as encoder and VGG-style decoder.

        This model builds a pyramid of intermediate feature maps from the encoder, and
        passes them to the decoder for semantic segmentation.

        Parameters
        ----------
        patch_size : tuple of int or int
            Shape of the input patch (D, H, W). The encoder will alwayse use an int but the config will always send a tuple ¯\\_(ツ)_/¯.
        num_pools : list of int, default=[5, 5, 5]
            Number of pooling steps per spatial dimension.
        num_classes : int, default=1
            Number of output segmentation classes.
        factor : int, default=32
            Base scaling factor for the decoder channels.
        encoder_ckpt : str or None, optional
            Path to a pretrained encoder checkpoint.
        model_ckpt : str or None, optional
            Path to a full model checkpoint.
        use_deep : bool, default=True
            Whether to use deep supervision in the decoder.
        in_planes : int, default=1
            Number of input channels.
        """
        super(EffUNet, self).__init__()

        pyramid={                       # efficientnet b4
        0: ['_bn0'],                    
        1: ['_blocks', '1', '_bn2'],   
        2: ['_blocks', '5', '_bn2'],  
        3: ['_blocks', '9', '_bn2'],    
        4: ['_blocks', '21', '_bn2'],  
        5: ['_blocks', '31', '_bn2'],  
        }

        blocks_args, global_params = efficientnet3d(
            width_coefficient=1.4, # efficientnet b4
            depth_coefficient=1.8,
            dropout_rate=0.4,
            drop_connect_rate=0.2,
            image_size=patch_size,
            include_top=False
        )
        self.encoder = EfficientNet3D(
            blocks_args,
            global_params, 
            in_channels=in_planes, 
            num_pools=num_pools,
        )

        # load encoder if needed
        if encoder_ckpt is not None:
            print("Load encoder weights from", encoder_ckpt)
            if torch.cuda.is_available():
                self.encoder.cuda()
            elif torch.backends.mps.is_available():
                self.encoder.to('mps')
            ckpt = torch.load(encoder_ckpt)
            if 'model' in ckpt.keys():
                # remove `module.` prefix
                state_dict = {k.replace("module.", ""): v for k, v in ckpt['model'].items()} 
                # remove `0.` prefix induced by the sequential wrapper
                state_dict = {k.replace("0.layers", "layers"): v for k, v in state_dict.items()}  
                print(self.encoder.load_state_dict(state_dict, strict=False))
            elif 'teacher' in ckpt.keys():
                # remove `module.` prefix
                state_dict = {k.replace("module.", ""): v for k, v in ckpt['teacher'].items()}  
                # remove `backbone.` prefix induced by multicrop wrapper
                state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
                print(self.encoder.load_state_dict(state_dict, strict=False))
            else:
                print("[Warning] the following encoder couldn't be loaded, wrong key:", encoder_ckpt)
        
        self.pyramid = get_pyramid(self.encoder, pyramid) # only the first five elements of the list are used

        # hook the pyramid
        self.down = {}
        for i in range(len(self.pyramid)):
            self.pyramid[i].register_forward_hook(self.get_activation(i))

        self.decoder = VGGDecoder(
            EncoderBlock,
            # SmallEncoderBlock,
            num_pools=num_pools,
            num_classes=num_classes,
            factor_e=[get_outfmaps(l) for l in self.pyramid][::-1],
            factor_d=[get_outfmaps(l)*2 for l in self.pyramid][::-1][1:-1]+[factor],
            use_deep=use_deep,
            )


        if model_ckpt is not None:
            print("Load model weights from", model_ckpt)
            if torch.cuda.is_available():
                self.cuda()
            elif torch.backends.mps.is_available():
                self.to('mps')
            ckpt = torch.load(model_ckpt)
            if 'encoder.last_layer.weight' in ckpt['model'].keys():
                del ckpt['model']['encoder.last_layer.weight']
            self.load_state_dict(ckpt['model'])

    def freeze_encoder(self, freeze:bool=True)->None:
        """
        Freeze or unfreeze the encoder's weights.

        Parameters
        ----------
        freeze : bool, optional
            If True, disables gradient computation for encoder parameters.

        Returns
        -------
        None
        """
        if freeze:
            print("Freezing encoder weights...")
        else:
            print("Unfreezing encoder weights...")
        for l in self.encoder.parameters(): 
            l.requires_grad = not freeze
    
    def unfreeze_encoder(self)->None:
        """Shortcut for unfreezing the encoder."""
        self.freeze_encoder(False)

    def get_activation(self, name:str)->Callable:
        """
        Create a forward hook for capturing activations.

        Parameters
        ----------
        name : int
            Index of the pyramid level to assign the activation to.

        Returns
        -------
        function
            A forward hook function.
        """
        def hook(model, input, output):
            self.down[name] = output
        return hook

    def forward(self, x:torch.Tensor)->torch.Tensor: 
        """
        Forward pass of the model.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (N, C, D, H, W).

        Returns
        -------
        torch.Tensor
            Output segmentation map of shape (N, num_classes, D, H, W).
        """
        self.encoder(x)
        out = self.decoder(list(self.down.values()))
        return out

#---------------------------------------------------------------------------
