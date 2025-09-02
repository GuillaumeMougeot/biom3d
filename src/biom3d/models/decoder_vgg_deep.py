"""3D VGG decoder, with deep supervision (each decoder level has an output)."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import numpy as np

from biom3d.utils import convert_num_pools

#---------------------------------------------------------------------------
# 3D Resnet decoder

def _weights_init(m:nn.Module)->None:
    """
    Initialize weights of convolutional and linear layers using Kaiming normal initialization.

    Parameters
    ----------
    m : nn.Module
        A PyTorch module. If it's an instance of `nn.Conv3d` or `nn.Linear`, its weights will be initialized.

    Returns
    -------
    None
    """
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv3d):
        init.kaiming_normal_(m.weight)

class DecoderBlock(nn.Module):
    """
    A decoder block consisting of an upsampling operation followed by an encoder block.

    This block upsamples the lower-resolution feature map, concatenates it with a skip connection
    from the encoder, and processes the result through a residual encoder block.

    :ivar int expansion: Expansion factor of the encoder block (default is 1).
    :ivar nn.ConvTranspose3d up: Transposed convolution used for upsampling.
    :ivar nn.Module encoder_block: Encoder block applied after concatenation.
    """

    expansion:int = 1
    up: nn.ConvTranspose3d
    encoder_block:nn.Module

    def __init__(self, 
        block:type[nn.Module],
        in_planes_low:int,  # in_planes is the depth size after the concatenation
        in_planes_high:int,  # in_planes is the depth size after the concatenation
        planes:int,     # the depth size of the output 
        stride:list[int], # for the upconv
        # option='A'# option for the upsampling (A: use upsamble; B: use convtranspose NOT IMPLEMENTED!)
        ):
        """
        Initialize a decoder block consisting of an upsampling operation followed by an encoder block.

        Parameters
        ----------
        block : type[nn.Module]
            Class of the residual block used to build encoder/decoder layers (e.g., EncoderBlock).
        in_planes_low : int
            Number of input channels from the low-resolution feature map.
        in_planes_high : int
            Number of channels from the high-resolution skip connection.
        planes : int
            Number of output channels after the encoder block.
        stride : list of int
            Stride for the transposed convolution (upsampling factor).
        """
        super(DecoderBlock, self).__init__()

        # if option == 'A': 
        #     self.up = nn.Upsample(scale_factor=2, mode='trilinear') # use bilinear but can be changed...
        # elif option == 'B':
        self.up = nn.ConvTranspose3d(in_planes_low, in_planes_high, kernel_size=stride, stride=stride, bias=False)

        self.encoder_block = block(
            in_planes=in_planes_high*2,
            planes=planes,
            stride=1,
            )

    def forward(self, x:list[torch.Tensor])->torch.Tensor: # x is a list of two inputs [low_res, high_res]
        """
        Forward pass of the decoder block.

        Parameters
        ----------
        x : list of torch.Tensor
            A pair [low_res, high_res] of feature maps to be merged.

        Returns
        -------
        torch.Tensor
            Output of the encoder block after upsampling and concatenation.
        """
        low, high = x
        low = self.up(low)
        out = torch.cat([low,high],dim=1)
        out = self.encoder_block(out)
        return out

class VGGDecoder(nn.Module):
    """
    A VGG-style decoder with optional deep supervision and intermediate embeddings.

    This decoder reconstructs feature maps by progressively upsampling and fusing skip connections
    from an encoder. It supports multi-scale supervision and embedding output.

    :ivar bool use_deep: If True, enable deep supervision (multi-level outputs).
    :ivar bool use_emb: If True, only return the intermediate embedding from the third decoder stage.
    :ivar list[list[int]] strides: List of strides (upsampling factors) per decoder stage.
    :ivar nn.ModuleList layers: List of DecoderBlocks composing the decoder.
    :ivar nn.ModuleList convs: List of 1×1 convolutions applied after each decoder stage (for supervision).
    """

    def __init__(
        self, 
        block:type[nn.Module], 
        num_pools:list[int], 
        factor_e:int|list[int] = 32, 
        factor_d:int|list[int] = 32, 
        flip_strides:bool = False, 
        num_classes:int = 1,
        use_deep:bool=True, 
        use_emb:bool=False, 
        roll_strides:bool = True, 
        ):
        """
        Initialize the decoder architecture.

        Parameters
        ----------
        block : type[nn.Module]
            Class of the residual block used to build encoder/decoder layers (e.g., EncoderBlock).
        num_pools : list of int
            Number of pooling operations at each encoder stage.
        factor_e : int or list of int, default=32
            Base or per-layer depth factor for encoder feature maps.
        factor_d : int or list of int, default=32
            Base or per-layer depth factor for decoder feature maps.
        flip_strides : bool, default=False
            Whether to reverse the order of upsampling strides. Flipped strides creates larger feature maps.
        num_classes : int, default=1
            Number of output channels (e.g. segmentation classes).
        use_deep : bool, default=True
            If True, enables deep supervision at multiple decoder levels.
        use_emb : bool, default=False
            If True, return the third decoder output as an embedding.
        roll_strides : bool, default=True
            Legacy support for reversing encoder stride order (for older models, before commit f2ac9ee (August 2023)).
        """
        super(VGGDecoder, self).__init__()

        self.use_deep = use_deep
        self.use_emb = use_emb

        # encoder pyramid planes/feature maps
        max_num_pools = max(num_pools)+1
        if isinstance(factor_e,int):
            in_planes = [factor_e * i for i in [10,10,8,4,2,1]][-max_num_pools:]
        elif isinstance(factor_e,list):
            in_planes = factor_e[-max_num_pools:]
        else:
            print("[Error] factor_e has the wrong type {}".format(type(factor_e)))
        
        in_planes_high = in_planes[1:]


        # decoder planes/feature maps
        if isinstance(factor_d,int):
            planes = [factor_d * i for i in [10,8,4,2,1]][-max_num_pools+1:]
        elif isinstance(factor_d,list):
            planes = factor_d
        else:
            print("[Error] factor_d has the wrong type {}".format(type(factor_d)))

        in_planes_low = [in_planes[0]]+planes[:-1]
            
        # computes the strides for the scale factors
        self.strides = convert_num_pools(num_pools=num_pools,roll_strides=roll_strides)
        # if the encoder strides are flipped, the decoder strides are not
        if not flip_strides: self.strides = np.flip(self.strides, axis=0).tolist()

        # layer definition
        self.layers = []
        self.convs = []
        for i in range(max_num_pools-1):
            self.layers += [self._make_layer(
                block,
                in_planes_low=in_planes_low[i],
                in_planes_high=in_planes_high[i],
                planes=planes[i], 
                stride=self.strides[i],
                num_blocks=1,
                )]
            self.convs += [nn.Conv3d(planes[i], num_classes, kernel_size=1, stride=1, padding=0, bias=False)]
        
        # the lines below are required to register the module parameters 
        self.layers = nn.ModuleList(self.layers)
        self.convs = nn.ModuleList(self.convs)

        self.apply(_weights_init)

    def _make_layer(self, 
        block:type[nn.Module],     
        in_planes_low:int,  
        in_planes_high:int,  
        planes:int,      
        stride:list[int],     
        num_blocks:int
        )->nn.Sequential:
        """
        Create a sequential layer composed of a DecoderBlock followed by encoder blocks.

        This function builds a composite decoder stage. It first upsamples and fuses encoder features
        using `DecoderBlock`, then adds more encoder blocks..

        Parameters
        ----------
        block : type[nn.Module]
            Class of the encoder-style residual block used after the initial DecoderBlock.
        in_planes_low : int
            Number of input channels from the lower-resolution feature map.
        in_planes_high : int
            Number of input channels from the higher-resolution feature map (skip connection).
        planes : int
            Number of output channels after the decoder stage.
        stride : list of int
            Stride (upsampling factor) for the transposed convolution.
        num_blocks : int
            Number of residual blocks to apply in total (≥1). The first is wrapped in a DecoderBlock.

        Returns
        -------
        nn.Sequential
            A sequential module containing the DecoderBlock and additional residual blocks.
        """
        layers = []
        layers.append(DecoderBlock(block, in_planes_low, in_planes_high, planes, stride))
        for _ in range(num_blocks-1):
            layers.append(block(planes, planes, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x:list[torch.Tensor]): 
        """
        Forward pass through the decoder.

        Parameters
        ----------
        x : list of torch.Tensor
            List of feature maps from the encoder (length depends on number of stages, but it should be 6).

        Returns
        -------
        out: torch.Tensor or list of torch.Tensor
            Final prediction map or list of maps (if deep supervision is enabled).
            If `use_emb` is True, returns only the intermediate embedding tensor.
        """
        deep_out = []
        for i in range(len(self.layers)):
            inputs = x[-1] if i==0 else out
            out = self.layers[i]([inputs, x[-2-i]])

            if i>=2 and self.use_deep: # deep supervision
                tmp = self.convs[i](out)
                deep_out += [F.interpolate(tmp, size=x[0].shape[2:], mode='trilinear')]
            # if i is the antipenultimate layer
            elif i==(len(self.layers)-3) and self.use_emb: 
                return self.convs[i](out)
        out = self.convs[-1](out)
        if self.use_deep: out = deep_out+[out]
        return out

#---------------------------------------------------------------------------