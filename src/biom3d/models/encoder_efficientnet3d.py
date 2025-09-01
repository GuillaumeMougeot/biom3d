"""
This file contains helper functions for building the model and for loading model parameters. These helper functions are built to mirror those in the official TensorFlow implementation.

3D EfficientNet adapted from:
https://github.com/shijianjian/EfficientNet-PyTorch-3D

Usage example:

.. code-block:: python

    model = EfficientNet3D.from_name("efficientnet-b1", override_params={'include_top': False}, in_channels=1)
    model.cuda()  # On CUDA machine
    model.to('mps')  # On Apple Silicon

List of pyramid layers:

.. code-block:: python

    {
        0: ['_conv_stem'],              # 100
        1: ['_blocks', '1', '_bn0'],    # 50
        2: ['_blocks', '3', '_bn0'],    # 25
        3: ['_blocks', '5', '_bn0'],    # 12
        4: ['_blocks', '11', '_bn0'],   # 6
        5: ['_bn1']                     # 3
    }

"""


import re
import math
import collections
from typing import Optional
import torch
from torch import nn
from torch.nn import functional as F
import numpy as np

########################################################################
############### HELPERS FUNCTIONS FOR MODEL ARCHITECTURE ###############
########################################################################


# Parameters for the entire model (stem, all blocks, and head)
GlobalParams = collections.namedtuple('GlobalParams', [
    'batch_norm_momentum', 'batch_norm_epsilon', 'dropout_rate',
    'num_classes', 'width_coefficient', 'depth_coefficient',
    'depth_divisor', 'min_depth', 'drop_connect_rate', 'image_size', 'include_top'])
"""
Global model parameters used across the entire network (stem, blocks, and head).

Parameters
----------
batch_norm_momentum : float
    Momentum for batch normalization layers.
batch_norm_epsilon : float
    Small epsilon value for numerical stability in batch norm.
dropout_rate : float
    Dropout rate used before the classifier head.
num_classes : int
    Number of output classes for classification.
width_coefficient : float
    Width scaling coefficient for number of filters.
depth_coefficient : float
    Depth scaling coefficient for number of block repeats.
depth_divisor : int
    Divisor to ensure the number of filters is divisible by this value.
min_depth : int
    Minimum depth (number of filters).
drop_connect_rate : float
    Drop connect probability for stochastic depth.
image_size : int
    Input image size.
include_top : bool
    Whether to include the fully-connected classifier head.
"""


# Parameters for an individual model block
BlockArgs = collections.namedtuple('BlockArgs', [
    'kernel_size', 'num_repeat', 'input_filters', 'output_filters',
    'expand_ratio', 'id_skip', 'stride', 'se_ratio'])
"""
Named tuple describing the parameters for a single model block.

Parameters
----------
kernel_size : int
    Size of the convolution kernel.
num_repeat : int
    Number of block repeats.
input_filters : int
    Number of input filters.
output_filters : int
    Number of output filters.
expand_ratio : int
    Expansion ratio for internal layers.
id_skip : bool
    Whether to use skip connections.
stride : int
    Stride size.
se_ratio : float or None
    Squeeze-and-excitation ratio.
"""

# Change namedtuple defaults
GlobalParams.__new__.__defaults__ = (None,) * len(GlobalParams._fields)
BlockArgs.__new__.__defaults__ = (None,) * len(BlockArgs._fields)


class SwishImplementation(torch.autograd.Function):
    """
    Memory-efficient implementation of the Swish activation function.

    Swish function: x * sigmoid(x)
    """

    @staticmethod
    def forward(ctx, i:torch.Tensor)->torch.Tensor:
        """
        Forward pass for Swish activation.

        Parameters
        ----------
        ctx : Context object
            Used to stash information for backward computation.
        i : torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor
            Output tensor after applying Swish.
        """
        result = i * torch.sigmoid(i)
        ctx.save_for_backward(i)
        return result

    @staticmethod
    def backward(ctx, grad_output:torch.Tensor)->torch.Tensor:
        """
        Backward pass for Swish activation.

        Parameters
        ----------
        ctx : Context object
            Contains saved tensors from forward pass.
        grad_output : torch.Tensor
            Gradient of the loss with respect to the output.

        Returns
        -------
        torch.Tensor
            Gradient of the loss with respect to the input.
        """
        i = ctx.saved_variables[0]
        sigmoid_i = torch.sigmoid(i)
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))


class MemoryEfficientSwish(nn.Module):
    """Module wrapping the memory-efficient Swish activation function."""

    def forward(self, x:torch.Tensor)->torch.Tensor:
        """
        Apply the memory-efficient Swish activation.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor
            Output tensor after Swish activation.
        """
        return SwishImplementation.apply(x)

class Swish(nn.Module):
    """Standard Swish activation function module."""
     
    def forward(self, x:torch.Tensor)->torch.Tensor:
        """
        Apply Swish activation function: x * sigmoid(x).

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor
            Output tensor after Swish activation.
        """
        return x * torch.sigmoid(x)


def round_filters(filters:int, global_params:GlobalParams)->int:
    """
    Calculate and round number of filters based on depth multiplier.

    Parameters
    ----------
    filters : int
        Original number of filters.
    global_params : GlobalParams
        Global parameters containing width_coefficient, depth_divisor, and min_depth.

    Returns
    -------
    int
        Rounded (floor) number of filters adjusted by width coefficient and divisor.
    """
    multiplier = global_params.width_coefficient
    if not multiplier:
        return filters
    divisor = global_params.depth_divisor
    min_depth = global_params.min_depth
    filters *= multiplier
    min_depth = min_depth or divisor
    new_filters = max(min_depth, int(filters + divisor / 2) // divisor * divisor)
    if new_filters < 0.9 * filters:  # prevent rounding by more than 10%
        new_filters += divisor
    return int(new_filters)


def round_repeats(repeats:int, global_params:GlobalParams)->int:
    """
    Round the number of block repeats ased on depth multiplier..

    Parameters
    ----------
    repeats : int
        Original number of repeats.
    global_params : GlobalParams
        Global parameters containing depth_coefficient.

    Returns
    -------
    int
        Rounded (ceil) number of repeats adjusted by depth coefficient.
    """
    multiplier = global_params.depth_coefficient
    if not multiplier:
        return repeats
    return int(math.ceil(multiplier * repeats))


def drop_connect(inputs:torch.Tensor, p:float, training:bool)->torch.Tensor:
    """
    Apply drop connect (stochastic depth) regularization.

    Parameters
    ----------
    inputs : torch.Tensor
        Input tensor of shape (batch_size, ...).
    p : float
        Drop connect probability.
    training : bool
        Whether the model is in training mode.

    Returns
    -------
    torch.Tensor
        Tensor after applying drop connect.
    """
    if not training: return inputs
    batch_size = inputs.shape[0]
    keep_prob = 1 - p
    random_tensor = keep_prob
    random_tensor += torch.rand([batch_size, 1, 1, 1, 1], dtype=inputs.dtype, device=inputs.device)
    binary_tensor = torch.floor(random_tensor)
    output = inputs / keep_prob * binary_tensor
    return output

########################################################################
############## HELPERS FUNCTIONS FOR LOADING MODEL PARAMS ##############
########################################################################


def efficientnet_params(model_name:str)->tuple[float,float,int,float]:
    """
    Map EfficientNet model name to parameter coefficients.

    Parameters
    ----------
    model_name : str
        Name of the EfficientNet model.

    Returns
    -------
    tuple of (float, float, int, float)
        A tuple with width_coefficient, depth_coefficient, image_size, dropout_rate.
    """
    params_dict = {
        # Coefficients:   width,depth,res,dropout
        'efficientnet-b0': (1.0, 1.0, 224, 0.2),
        'efficientnet-b1': (1.0, 1.1, 240, 0.2),
        'efficientnet-b2': (1.1, 1.2, 260, 0.3),
        'efficientnet-b3': (1.2, 1.4, 300, 0.3),
        'efficientnet-b4': (1.4, 1.8, 380, 0.4),
        'efficientnet-b5': (1.6, 2.2, 456, 0.4),
        'efficientnet-b6': (1.8, 2.6, 528, 0.5),
        'efficientnet-b7': (2.0, 3.1, 600, 0.5),
        'efficientnet-b8': (2.2, 3.6, 672, 0.5),
        'efficientnet-l2': (4.3, 5.3, 800, 0.5),
    }
    return params_dict[model_name]


class BlockDecoder(object):
    """
    Block Decoder for EfficientNet blocks. From Tensorflow repository.

    Provides utilities to encode/decode block configuration strings
    into BlockArgs namedtuples for better readability and processing.
    """

    @staticmethod
    def _decode_block_string(block_string:str)->BlockArgs:
        """
        Decode a single block string to a BlockArgs namedtuple.

        Parameters
        ----------
        block_string : str
            String describing the block parameters (e.g. 'r1_k3_s222_e1_i32_o16_se0.25').

        Returns
        -------
        BlockArgs
            Namedtuple describing the block parameters.
        """
        assert isinstance(block_string, str)

        ops = block_string.split('_')
        options = {}
        for op in ops:
            splits = re.split(r'(\d.*)', op)
            if len(splits) >= 2:
                key, value = splits[:2]
                options[key] = value

        # Check stride
        assert (('s' in options and len(options['s']) == 1) or
                (len(options['s']) == 3 and options['s'][0] == options['s'][1] == options['s'][2]))

        return BlockArgs(
            kernel_size=int(options['k']),
            num_repeat=int(options['r']),
            input_filters=int(options['i']),
            output_filters=int(options['o']),
            expand_ratio=int(options['e']),
            id_skip=('noskip' not in block_string),
            se_ratio=float(options['se']) if 'se' in options else None,
            stride=[int(options['s'][0])])

    @staticmethod
    def _encode_block_string(block:BlockArgs)->str:
        """
        Encode a BlockArgs namedtuple back to a string.

        Parameters
        ----------
        block : BlockArgs
            BlockArgs namedtuple to encode.

        Returns
        -------
        str
            Encoded string describing the block.
        """
        args = [
            'r%d' % block.num_repeat,
            'k%d' % block.kernel_size,
            's%d%d%d' % (block.strides[0], block.strides[1], block.strides[2]),
            'e%s' % block.expand_ratio,
            'i%d' % block.input_filters,
            'o%d' % block.output_filters
        ]
        if 0 < block.se_ratio <= 1:
            args.append('se%s' % block.se_ratio)
        if block.id_skip is False:
            args.append('noskip')
        return '_'.join(args)

    @staticmethod
    def decode(string_list:list[str])->list[BlockArgs]:
        """
        Decode a list of block strings into a list of BlockArgs.

        Parameters
        ----------
        string_list : list of str
            List of block description strings.

        Returns
        -------
        list of BlockArgs
            List of decoded block arguments.
        """
        assert isinstance(string_list, list)
        blocks_args = []
        for block_string in string_list:
            blocks_args.append(BlockDecoder._decode_block_string(block_string))
        return blocks_args

    @staticmethod
    def encode(blocks_args:list[BlockArgs])->list[str]:
        """
        Encode a list of BlockArgs into a list of strings.

        Parameters
        ----------
        blocks_args : list of BlockArgs
            List of block arguments to encode.

        Returns
        -------
        list of str
            List of encoded block description strings.
        """
        block_strings = []
        for block in blocks_args:
            block_strings.append(BlockDecoder._encode_block_string(block))
        return block_strings


def efficientnet3d(width_coefficient:Optional[float]=None, 
                   depth_coefficient:Optional[float]=None, 
                   dropout_rate:float=0.2,
                   drop_connect_rate:float=0.2, 
                   image_size:Optional[int]=None, 
                   num_classes:int=1000, 
                   include_top:bool=True,
                   )->tuple[BlockArgs,GlobalParams]:
    """
    Create EfficientNet3D block arguments and global parameters.

    Parameters
    ----------
    width_coefficient : float, optional
        Width multiplier to scale number of filters.
    depth_coefficient : float, optional
        Depth multiplier to scale number of layers.
    dropout_rate : float, default=0.2
        Dropout rate before the classifier.
    drop_connect_rate : float, default=0.2
        Drop connect rate for stochastic depth.
    image_size : int, optional
        Input image size (default is None).
    num_classes : int, default=1000
        Number of output classes (default is 1000).
    include_top : bool, default=True
        Whether to include the classification head (default is True).

    Returns
    -------
    blocks_args : list of BlockArgs
        List of block arguments describing the network architecture.
    global_params : GlobalParams
        Namedtuple of global model parameters.
    """
    blocks_args = [
        'r1_k3_s222_e1_i32_o16_se0.25', 'r2_k3_s222_e6_i16_o24_se0.25',
        'r2_k5_s222_e6_i24_o40_se0.25', 'r3_k3_s222_e6_i40_o80_se0.25',
        'r3_k5_s111_e6_i80_o112_se0.25', 'r4_k5_s222_e6_i112_o192_se0.25',
        'r1_k3_s111_e6_i192_o320_se0.25',
    ]
    blocks_args = BlockDecoder.decode(blocks_args)

    global_params = GlobalParams(
        batch_norm_momentum=0.99,
        batch_norm_epsilon=1e-3,
        dropout_rate=dropout_rate,
        drop_connect_rate=drop_connect_rate,
        num_classes=num_classes,
        width_coefficient=width_coefficient,
        depth_coefficient=depth_coefficient,
        depth_divisor=8,
        min_depth=None,
        image_size=image_size,
        include_top=include_top,
    )

    return blocks_args, global_params


def get_model_params(model_name:str, override_params:Optional[dict])->tuple[BlockArgs,GlobalParams]:
    """
    Retrieve EfficientNet block args and global parameters by model name.

    Parameters
    ----------
    model_name : str
        EfficientNet model name (e.g., 'efficientnet-b0').
    override_params : dict or None
        Dictionary to override default global parameters.

    Raises
    ------
    NotImplementedError
        If model name doesn't start with 'efficientnet'
    ValueError
        If override_params has field not in GlobalParams

    Returns
    -------
    blocks_args : list of BlockArgs
        List of block arguments.
    global_params : GlobalParams
        Namedtuple of global parameters.
    """
    if model_name.startswith('efficientnet'):
        w, d, s, p = efficientnet_params(model_name)
        # note: all models have drop connect rate = 0.2
        blocks_args, global_params = efficientnet3d(
            width_coefficient=w, depth_coefficient=d, dropout_rate=p, image_size=s)
    else:
        raise NotImplementedError('model name is not pre-defined: %s' % model_name)
    if override_params:
        # ValueError will be raised here if override_params has fields not included in global_params.
        global_params = global_params._replace(**override_params)
    return blocks_args, global_params

class MBConvBlock3D(nn.Module):
    """
    Mobile Inverted Residual Bottleneck Block.

    :ivar bool has_se: Whether the block contains a Squeeze and Excitation layer.
    """

    def __init__(self, block_args:BlockArgs, global_params:GlobalParams):
        """
        Mobile Inverted Residual Bottleneck Block (3D).

        Parameters
        ----------
        block_args : namedtuple
            BlockArgs namedtuple containing block configuration.
        global_params : namedtuple
            GlobalParams namedtuple containing global model parameters.
        """
        super().__init__()
        self._block_args = block_args
        self._bn_mom = 1 - global_params.batch_norm_momentum
        self._bn_eps = global_params.batch_norm_epsilon
        self.has_se = (self._block_args.se_ratio is not None) and (0 < self._block_args.se_ratio <= 1)
        self.id_skip = block_args.id_skip  # skip connection and drop connect

        # Get static or dynamic convolution depending on image size
        Conv3d = nn.Conv3d

        # Expansion phase
        inp = self._block_args.input_filters  # number of input channels
        oup = self._block_args.input_filters * self._block_args.expand_ratio  # number of output channels
        if self._block_args.expand_ratio != 1:
            self._expand_conv = Conv3d(in_channels=inp, out_channels=oup, kernel_size=1, bias=False)
            self._bn0 = nn.InstanceNorm3d(num_features=oup)

        # Depthwise convolution phase
        k = self._block_args.kernel_size
        s = self._block_args.stride
        self._depthwise_conv = Conv3d(
            in_channels=oup, out_channels=oup, groups=oup,  # groups makes it depthwise
            kernel_size=k, stride=s, bias=False, padding=np.array(k)//2)
        self._bn1 = nn.InstanceNorm3d(num_features=oup)

        # Squeeze and Excitation layer, if desired
        if self.has_se:
            num_squeezed_channels = max(1, int(self._block_args.input_filters * self._block_args.se_ratio))
            self._se_reduce = Conv3d(in_channels=oup, out_channels=num_squeezed_channels, kernel_size=1)
            self._se_expand = Conv3d(in_channels=num_squeezed_channels, out_channels=oup, kernel_size=1)

        # Output phase
        final_oup = self._block_args.output_filters
        self._project_conv = Conv3d(in_channels=oup, out_channels=final_oup, kernel_size=1, bias=False)
        self._bn2 = nn.InstanceNorm3d(num_features=final_oup)
        self._swish = MemoryEfficientSwish()

    def forward(self, inputs:torch.Tensor, drop_connect_rate:Optional[float]=None)->torch.Tensor:
        """
        Forward pass through the MBConv block.

        Parameters
        ----------
        inputs : torch.Tensor
            Input tensor to the block.
        drop_connect_rate : float, optional
            Drop connect probability (between 0 and 1). Default is None.

        Returns
        -------
        torch.Tensor
            Output tensor after processing through the block.
        """
        # Expansion and Depthwise Convolution
        x = inputs
        if self._block_args.expand_ratio != 1:
            x = self._swish(self._bn0(self._expand_conv(inputs)))
        x = self._swish(self._bn1(self._depthwise_conv(x)))

        # Squeeze and Excitation
        if self.has_se:
            x_squeezed = F.adaptive_avg_pool3d(x, 1)
            x_squeezed = self._se_expand(self._swish(self._se_reduce(x_squeezed)))
            x = torch.sigmoid(x_squeezed) * x

        x = self._bn2(self._project_conv(x))

        # Skip connection and drop connect
        input_filters, output_filters = self._block_args.input_filters, self._block_args.output_filters
        if self.id_skip and self._block_args.stride == 1 and input_filters == output_filters:
            if drop_connect_rate:
                x = drop_connect(x, p=drop_connect_rate, training=self.training)
            x = x + inputs  # skip connection
        return x

    def set_swish(self, memory_efficient:bool=True)->None:
        """
        Set the Swish activation function implementation.

        Parameters
        ----------
        memory_efficient : bool, optional
            If True, use memory-efficient Swish (suitable for training).
            If False, use standard Swish (suitable for export). Default is True.

        Returns
        -------
        None
        """
        self._swish = MemoryEfficientSwish() if memory_efficient else Swish()


class EfficientNet3D(nn.Module):
    """
    An EfficientNet model. Most easily loaded with the .from_name or .from_pretrained methods.

    :ivar nn.ModuleList _blocks: List of MBConvBlock3D blocks making up the network.
    :ivar nn.Conv3d _conv_stem: Initial convolutional stem layer.
    :ivar nn.Conv3d _conv_head: Final convolutional head layer before classifier.
    :ivar nn.InstanceNorm3d _bn0: Batch normalization after the stem.
    :ivar nn.BatchNorm3d _bn1: Batch normalization after the head.
    :ivar nn.AdaptiveAvgPool3d _avg_pooling: Global average pooling layer.
    :ivar nn.Dropout _dropout: Dropout layer before the classifier.
    :ivar nn.Linear _fc: Fully connected linear layer for classification.
    :ivar nn.Module _swish: Swish activation function module.

    Examples
    --------
    >>> model = EfficientNet3D.from_name('efficientnet-b0')
    >>> x = torch.randn(1, 3, 224, 224, 224)
    >>> logits = model(x)

    """

    def __init__(self, blocks_args:BlockArgs, 
                 global_params:Optional[GlobalParams]=None, 
                 in_channels:int=3, 
                 num_pools:list[int]=[5,5,5], 
                 first_stride:list[int]=[1,1,1]):
        """
        Efficientnet 3D model implementation.

        Parameters
        ----------
        blocks_args : list of BlockArgs
            List of block argument namedtuples to construct the network blocks.
        global_params : namedtuple
            Global parameters shared across blocks (e.g., dropout rate, batch norm params).
        in_channels : int, optional
            Number of input channels. Default is 3.
        num_pools : list of int, optional
            List defining the number of pooling layers per dimension. Default is [5, 5, 5].
        first_stride : list of int, optional
            Stride size of the first convolution layer in each spatial dimension. Default is [1, 1, 1].

        Raises
        ------
        AssertionError
            If block_args is not a list or empty
        """
        super().__init__()
        assert isinstance(blocks_args, list), 'blocks_args should be a list'
        assert len(blocks_args) > 0, 'block args must be greater than 0'
        self._global_params = global_params
        self._blocks_args = blocks_args

        # Get static or dynamic convolution depending on image size
        # Conv3d = get_same_padding_conv3d(image_size=global_params.image_size)
        Conv3d = nn.Conv3d

        # Batch norm parameters
        bn_mom = 1 - self._global_params.batch_norm_momentum
        bn_eps = self._global_params.batch_norm_epsilon

        # Stem
        out_channels = round_filters(32, self._global_params)  # number of output channels
        self._conv_stem = Conv3d(in_channels, out_channels, kernel_size=3, stride=first_stride, bias=False, padding=1)
        self._bn0 = nn.InstanceNorm3d(num_features=out_channels)

        # Set adaptive number of pools
        # for example: convert [3,5,5] into [[1 1 1],[1 2 2],[2 2 2],[2 2 2],[2 2 2],[1 2 2]]
        max_pool = max(num_pools)
        strides = []
        for i in range(len(num_pools)):
            st = np.ones(max_pool)
            num_zeros = max_pool-num_pools[i]
            for j in range(num_zeros):
                st[j]=0
            st=np.roll(st,-num_zeros//2)
            strides += [st]
        strides = np.array(strides).astype(int).T+1
        strides = strides.tolist()
        
        # Build blocks
        self._blocks = nn.ModuleList([])
        crt_stride = 0
        for block_args in self._blocks_args:

            # Update block input and output filters based on depth multiplier.
            block_args = block_args._replace(
                input_filters=round_filters(block_args.input_filters, self._global_params),
                output_filters=round_filters(block_args.output_filters, self._global_params),
                num_repeat=round_repeats(block_args.num_repeat, self._global_params),
            )
            
            if np.greater(block_args.stride,1):
                block_args = block_args._replace(stride=strides[crt_stride])
                crt_stride += 1

            # The first block needs to take care of stride and filter size increase.
            self._blocks.append(MBConvBlock3D(block_args, self._global_params))
            if block_args.num_repeat > 1:
                block_args = block_args._replace(input_filters=block_args.output_filters, stride=1)
            for _ in range(block_args.num_repeat - 1):
                self._blocks.append(MBConvBlock3D(block_args, self._global_params))

        # Head
        in_channels = block_args.output_filters  # output of final block
        out_channels = round_filters(1280, self._global_params)
        self._conv_head = Conv3d(in_channels, out_channels, kernel_size=1, bias=False)
        self._bn1 = nn.BatchNorm3d(num_features=out_channels, momentum=bn_mom, eps=bn_eps)

        # Final linear layer
        self._avg_pooling = nn.AdaptiveAvgPool3d(1)
        self._dropout = nn.Dropout(self._global_params.dropout_rate)
        self._fc = nn.Linear(out_channels, self._global_params.num_classes)
        self._swish = MemoryEfficientSwish()

    def set_swish(self, memory_efficient:bool=True)->None:
        """
        Set the Swish activation function implementation.

        Parameters
        ----------
        memory_efficient : bool, optional
            If True, use memory-efficient Swish (suitable for training).
            If False, use standard Swish (suitable for export). Default is True.

        Returns
        -------
        None
        """
        self._swish = MemoryEfficientSwish() if memory_efficient else Swish()
        for block in self._blocks:
            block.set_swish(memory_efficient)


    def extract_features(self, inputs:torch.Tensor)->torch.Tensor:
        """
        Extract features from inputs by forwarding through convolutional layers.

        Parameters
        ----------
        inputs : torch.Tensor
            Input tensor of shape (batch_size, channels, depth, height, width).

        Returns
        -------
        torch.Tensor
            Feature tensor after convolutional blocks and head.
        """
        # Stem
        x = self._swish(self._bn0(self._conv_stem(inputs)))

        # Blocks
        for idx, block in enumerate(self._blocks):
            drop_connect_rate = self._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self._blocks)
            x = block(x, drop_connect_rate=drop_connect_rate)

        # Head
        x = self._swish(self._bn1(self._conv_head(x)))

        return x

    def forward(self, inputs:torch.Tensor)->torch.Tensor:
        """
        Forward pass of the EfficientNet3D model. Apply final Linear layer.

        Parameters
        ----------
        inputs : torch.Tensor
            Input tensor of shape (batch_size, channels, depth, height, width).

        Returns
        -------
        torch.Tensor
            Logits tensor of shape (batch_size, num_classes) if include_top=True,
            otherwise feature tensor.
        """
        bs = inputs.size(0)
        # Convolution layers
        x = self.extract_features(inputs)
        
        if self._global_params.include_top:
            # Pooling and final linear layer
            x = self._avg_pooling(x)
            x = x.view(bs, -1)
            x = self._dropout(x)
            x = self._fc(x)
        return x

    @classmethod
    def from_name(cls:"EfficientNet3D", 
                  model_name:str, 
                  override_params:Optional[dict]=None, 
                  in_channels:int=3, 
                  **kwargs)->"EfficientNet3D":
        """
        Create an EfficientNet3D model from a predefined model name.

        Parameters
        ----------
        model_name : str
            Name of the EfficientNet model variant, e.g. 'efficientnet-b0'.
        override_params : dict or None, optional
            Dictionary to override global parameters.
        in_channels : int, optional
            Number of input channels.
        **kwargs : dict
            Additional keyword arguments passed to the constructor.

        Returns
        -------
        EfficientNet3D
            Constructed EfficientNet3D model.
        """
        cls._check_model_name_is_valid(model_name)
        blocks_args, global_params = get_model_params(model_name, override_params)
        return cls(blocks_args, global_params, in_channels, **kwargs)

    @classmethod
    def get_image_size(cls:"EfficientNet3D", model_name:str)->int:
        """
        Get the default input image size for a given EfficientNet model.

        Parameters
        ----------
        model_name : str
            EfficientNet model name.

        Returns
        -------
        int
            Default image size.
        """
        cls._check_model_name_is_valid(model_name)
        _, _, res, _ = efficientnet_params(model_name)
        return res

    @classmethod
    def _check_model_name_is_valid(cls:"EfficientNet3D", model_name:str)->None:
        """
        Check if the given model name is valid.

        Parameters
        ----------
        model_name : str
            Name of the model to validate.

        Raises
        ------
        ValueError
            If the model name is not valid.

        Returns
        -------
        None
        """
        valid_models = ['efficientnet-b'+str(i) for i in range(9)]
        if model_name not in valid_models:
            raise ValueError('model_name should be one of: ' + ', '.join(valid_models))

