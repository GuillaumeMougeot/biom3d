"""3D Resnet adapted from: https://github.com/akamaster/pytorch_resnet_cifar10."""

from typing import Callable, Iterable, Literal, Optional
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch import Tensor

from biom3d.utils import convert_num_pools

#---------------------------------------------------------------------------
# 3D Resnet encoder

def _weights_init(m:nn.Module)->None:
    """
    Initialize weights of the given module.

    Parameters
    ----------
    m : nn.Module
        Module to initialize.

    Returns
    -------
    None
    """
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, 0, 0.01)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

class LambdaLayer(nn.Module):
    """
    Applies a lambda function as a layer.

    :ivar callable lambd: lambda function to be applied in forward
    """

    def __init__(self, lambd:Callable):
        """
        Apply a lambda function as a layer.

        Parameters
        ----------
        lambd : callable
            Lambda function to apply in forward pass.
        """
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x:Tensor)->Tensor:
        """
        Forward pass applying the lambda function.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor
            Output after applying the lambda function.
        """
        return self.lambd(x)

class GlobalAvgPool3d(nn.Module):
    """
    Performs global average pooling over the last three dimensions.

    This layer averages the input tensor over the depth, height, and width dimensions.
    """

    def __init__(self):
        """
        Perform global average pooling over the last three dimensions.

        This layer averages the input tensor over the depth, height, and width dimensions.
        """
        super(GlobalAvgPool3d, self).__init__()

    def forward(self,x:Tensor)->Tensor:
        """
        Forward pass computing the global average pooling.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (N, C, D, H, W).

        Returns
        -------
        torch.Tensor
            Output tensor of shape (N, C) after global average pooling.
        """
        out = x.mean(dim=(-3,-2,-1))
        return out

class SmallEncoderBlock(nn.Module):
    """
    Small 3D encoder block with one convolution and optional normalization and activation.

    :ivar nn.Conv3d conv1: 3D convolution layer
    :ivar nn.InstanceNorm3d bn1: Instance normalization layer (only if is_last is False)
    :ivar bool is_last: indicates if this is the last block (no norm or activation)
    """

    conv1:nn.Conv3d
    is_last:bool

    def __init__(self, in_planes:int, planes:int, stride:int=1, option:Literal['A','B']='B', is_last:bool=False):
        """
        Small 3D encoder block with one convolution and optional normalization and activation.

        Parameters
        ----------
        in_planes : int
            Number of input channels.
        planes : int
            Number of output channels.
        stride : int, default=1
            Stride of the convolution.
        option : str, default='B'
            Option parameter used to initialize block (not used).
        is_last : bool, default=False
            If True, no normalization or activation is applied. 
        """
        super(SmallEncoderBlock, self).__init__()
        self.conv1 = nn.Conv3d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.is_last = is_last
        if not is_last:
            self.bn1 = nn.InstanceNorm3d(planes, affine=True)

    def forward(self, x:Tensor)->Tensor:
        """
        Forward pass through the block.

        Applies convolution, followed by instance normalization and LeakyReLU if not the last block.
        Otherwise, applies only convolution.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor
            Output tensor after block processing.
        """
        if not self.is_last:
            out = F.leaky_relu(self.bn1(self.conv1(x)), inplace=True)
        else:
            out = self.conv1(x)
        return out

class EncoderBlock(nn.Module):
    """
    A 3D convolutional encoder block with optional InstanceNorm and LeakyReLU activation.

    This block consists of two convolutional layers. The second normalization and activation
    are skipped if the block is marked as the last.

    :ivar nn.Conv3d conv1: First 3D convolution layer.
    :ivar nn.InstanceNorm3d bn1: Instance normalization applied after the first convolution.
    :ivar nn.Conv3d conv2: Second 3D convolution layer.
    :ivar bn2: Instance normalization applied after the second convolution (if not last block).
    :ivar bool is_last: Flag indicating if the block is the last in the sequence.
    """

    conv1:nn.Conv3d
    bn1:nn.InstanceNorm3d
    conv2:nn.Conv3d
    is_last:bool
    bn2: nn.InstanceNorm3d

    def __init__(self, in_planes:int, planes:int, stride:int=1, option:Literal['A','B']='B', is_last:bool=False):
        """
        3D convolutional encoder block with optional InstanceNorm and LeakyReLU activation.

        This block consists of two convolutional layers. The second normalization and activation
        are skipped if the block is marked as the last.

        Parameters
        ----------
        in_planes : int
            Number of input channels.
        planes : int
            Number of output channels.
        stride : int or tuple, default=1
            Stride for the first convolution layer.
        option : str, default='B'
            Not used in this implementation, placeholder for possible variants.
        is_last : bool, default=False
            Whether this block is the last one, which disables the second normalization and activation.
        """
        super(EncoderBlock, self).__init__()

        self.conv1 = nn.Conv3d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.InstanceNorm3d(planes, affine=True)
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)

        self.is_last = is_last
        if not is_last:
            self.bn2 = nn.InstanceNorm3d(planes, affine=True)

    def forward(self, x:Tensor)->Tensor:
        """
        Forward pass through the EncoderBlock.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (N, C, D, H, W).

        Returns
        -------
        torch.Tensor
            Output tensor after convolution, normalization, and activation.
        """
        out = F.leaky_relu(self.bn1(self.conv1(x)), inplace=True)
        if not self.is_last:
            out = F.leaky_relu(self.bn2(self.conv2(out)), inplace=True)
        else:
            out = self.conv2(out)
        return out

class VGGEncoder(nn.Module):
    """
    VGG-style 3D encoder composed of multiple EncoderBlocks.

    The architecture applies a sequence of blocks with progressively increasing number of channels,
    with configurable pooling and strides.

    :ivar int in_planes: Number of input channels to the current layer.
    :ivar bool use_emb: Whether embedding is used.
    :ivar bool use_head: Whether fully connected head is used.
    :ivar ModuleList layers: ModuleList containing the sequence of encoder layers.
    :ivar nn.Sequential head: Optional fully connected head for embedding (if use_head is True).
    """

    def __init__(self, 
        block:type[nn.Module], 
        num_pools:list[int], 
        factor:int = 32,
        first_stride:list[int]=[1,1,1], 
        flip_strides:bool = False, 
        use_emb:bool=False, 
        emb_dim:int=320,
        use_head:bool=False,
        patch_size:Optional[Iterable[int]] = None,
        in_planes:int = 1,
        roll_strides:bool = True,
        ): 
        """
        VGG-style 3D encoder composed of multiple EncoderBlocks.

        The architecture applies a sequence of blocks with progressively increasing number of channels,
        with configurable pooling and strides.

        Parameters
        ----------
        block : nn.Module
            Encoder block class to use (e.g. EncoderBlock).
        num_pools : list of int
            Number of pooling steps in each spatial dimension.
        factor : int, default=32
            Base factor for channel scaling.
        first_stride : list of int, default=[1,1,1]
            Stride for the first convolution layer. 
        flip_strides : bool, default=False
            Whether to flip the order of computed strides. Flipped strides creates larger feature maps.
        use_emb : bool, default=False
            Whether to use an embedding layer on top of the last encoder output.
        emb_dim : int, default=320
            Dimension of the embedding output.
        use_head : bool, default=False
            Whether to use a fully connected head after flattening.
        patch_size : iterable of int , optional
            Input patch size, needed if use_head is True.
        in_planes : int, default=1
            Number of input channels.
        roll_strides : bool, default=True
            Whether to roll strides when computing pooling (used for backward compatibility for models trained before commit f2ac9ee (August 2023)).
        """
        super(VGGEncoder, self).__init__()
        factors = [factor * i for i in [1,2,4,8,10,10,10]] # TODO: make this flexible to larger U-Net model?
        self.in_planes = in_planes
        self.use_emb=use_emb
        self.use_head = use_head

        # computes the strides
        # for example: convert [3,5,5] into [[1 1 1],[1 2 2],[2 2 2],[2 2 2],[2 2 2],[1 2 2]]
        strides = convert_num_pools(num_pools=num_pools,roll_strides=roll_strides)
        if flip_strides: strides = np.flip(strides, axis=0)
        strides = np.vstack(([first_stride],strides))
        strides = strides.tolist()

        # defines the network
        self.layers = []
        for i in range(max(num_pools)+1):
            self.layers += [self._make_layer(block, factors[i], num_blocks=1, stride=strides[i], is_last=(i==max(num_pools) and use_emb))]
        
        self.layers = nn.ModuleList(self.layers)

        if use_emb and  use_head:
            strides_ = (np.array(strides)).prod(axis=0)
            in_dim = (np.array(patch_size)/strides_).prod().astype(int)*in_planes*factors[-1]
            last_layer = nn.utils.weight_norm(nn.Linear(256, emb_dim, bias=False))
            # norm last layer
            last_layer.weight_g.data.fill_(1)
            last_layer.weight_g.requires_grad = False

            self.head = nn.Sequential(
                nn.Linear(in_dim, 2048),
                nn.GELU(),
                nn.Dropout(p=0.5),
                nn.Linear(2048, 2048),
                nn.GELU(),
                nn.Dropout(p=0.5),
                nn.Linear(2048, 256), # bottleneck
                last_layer,
            )

            

        self.apply(_weights_init)

    def _make_layer(self, 
                    block:nn.Module, 
                    planes:int, 
                    num_blocks:int, 
                    stride:list[int]|int, 
                    is_last:bool=False,
                    )->nn.Sequential:
        """
        Create a sequential layer composed of multiple blocks.

        Parameters
        ----------
        block : nn.Module
            The encoder block to use.
        planes : int
            Number of output channels.
        num_blocks : int
            Number of blocks to stack.
        stride : list or int
            Stride(s) to use for the first block.
        is_last : bool, default=False
            Whether this layer is the last one.

        Returns
        -------
        nn.Sequential
            Sequential container of blocks.
        """
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers += [block(self.in_planes, planes, stride, is_last=is_last)]
            self.in_planes = planes

        return nn.Sequential(*layers)

    def forward(self, x:Tensor, use_encoder:bool=False)->Tensor:
        """
        Forward pass through the VGGEncoder.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (N, C, D, H, W).
        use_encoder : bool, default=False
            Whether to apply the embedding head to the last output.

        Returns
        -------
        list of torch.Tensor or torch.Tensor
            List of intermediate feature maps if `use_emb` is False.
            If `use_emb` is True, returns the embedding vector (after flattening and head if use_encoder=True).
        """
        # stores the intermediate outputs
        out = []
        for i in range(len(self.layers)):
            inputs = x if i==0 else out[-1]
            out += [self.layers[i](inputs)]
            
        if self.use_emb:
            out = out[-1].view(out[-1].size(0), -1)
            if use_encoder:
                out = self.head(out)
            
        return out
#---------------------------------------------------------------------------

