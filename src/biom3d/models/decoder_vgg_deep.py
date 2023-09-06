#---------------------------------------------------------------------------
# 3D VGG decoder
# with deep supervision: meaning that each decoder level has an output
#---------------------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import numpy as np

from biom3d.utils import convert_num_pools

#---------------------------------------------------------------------------
# 3D Resnet decoder

def _weights_init(m):
    # classname = m.__class__.__name__
    # print(classname)
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv3d):
        init.kaiming_normal_(m.weight)

class DecoderBlock(nn.Module):
    expansion = 1

    def __init__(self, 
        block,
        in_planes_low,  # in_planes is the depth size after the concatenation
        in_planes_high,  # in_planes is the depth size after the concatenation
        planes,     # the depth size of the output 
        stride, # for the upconv
        # option='A'# option for the upsampling (A: use upsamble; B: use convtranspose NOT IMPLEMENTED!)
        ):

        super(DecoderBlock, self).__init__()

        # if option == 'A': 
        #     self.up = nn.Upsample(scale_factor=2, mode='trilinear') # use bilinear but can be changed...
        # elif option == 'B':
        self.up = nn.ConvTranspose3d(in_planes_low, in_planes_high, kernel_size=stride, stride=stride, bias=False)

        self.encoder_block = block(
            in_planes=in_planes_high*2,
            planes=planes,
            stride=1,
            # option='B',
            )

    def forward(self, x): # x is a list of two inputs [low_res, high_res]
        low, high = x
        low = self.up(low)
        # print("low",low.shape)
        # print("high",high.shape)
        out = torch.cat([low,high],dim=1)
        out = self.encoder_block(out)
        return out

class VGGDecoder(nn.Module):
    def __init__(
        self, 
        block, 
        num_pools, 
        factor_e = 32, # factor encoder
        factor_d = 32, # factor decoder
        flip_strides = False, # whether to invert strides order. Flipped strides creates larger feature maps.
        num_classes=1,
        use_deep=True,      # use deep supervision
        use_emb=False, # will only output the third level of the decoder
        ):
        super(VGGDecoder, self).__init__()

        self.use_deep = use_deep
        self.use_emb = use_emb

        # encoder pyramid planes/feature maps
        max_num_pools = max(num_pools)+1
        if type(factor_e)==int:
            in_planes = [factor_e * i for i in [10,10,8,4,2,1]][-max_num_pools:]
        elif type(factor_e)==list:
            in_planes = factor_e[-max_num_pools:]
        else:
            print("[Error] factor_e has the wrong type {}".format(type(factor_e)))
        
#         in_planes_low = in_planes[:-1]
        in_planes_high = in_planes[1:]


        # decoder planes/feature maps
        if type(factor_d)==int:
            planes = [factor_d * i for i in [10,8,4,2,1]][-max_num_pools+1:]
        elif type(factor_d)==list:
            planes = factor_d
        else:
            print("[Error] factor_d has the wrong type {}".format(type(factor_d)))

        in_planes_low = [in_planes[0]]+planes[:-1]
            
        # computes the strides for the scale factors
        self.strides = convert_num_pools(num_pools=num_pools)
        # if the encoder strides are flipped, the decoder strides are not
        if not flip_strides: self.strides = np.flip(self.strides, axis=0).tolist()

        # max_pool = max(num_pools)
        # strides = []
        # for i in range(len(num_pools)):
        #     st = np.ones(max_pool)
        #     num_zeros = max_pool-num_pools[i]
        #     for j in range(num_zeros):
        #         st[j]=0
        #     st=np.roll(st,-num_zeros//2)
        #     strides += [st]
        # strides = np.array(strides).astype(int).T+1
        # self.strides = np.flip(strides, axis=0).tolist()

        # computes the strides for the scale factors (old)
        # self.strides = []
        # num_pools_ = np.array(num_pools)
        # while num_pools_.sum()!=0:
        #     stride = (num_pools_>0).astype(int)
        #     num_pools_ -= stride
        #     self.strides += [list(stride+1)]
        # self.strides = np.flip(self.strides, axis=0).tolist()

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
        block,      # encoder block
        in_planes_low,  # depth size after concatenation
        in_planes_high,  
        planes,     # output depth size   
        stride,     # for the upconv
        num_blocks
        ):# number of encoder blocks
        layers = []
        layers.append(DecoderBlock(block, in_planes_low, in_planes_high, planes, stride))
        # self.in_planes = planes * block.expansion
        for _ in range(num_blocks-1):
            layers.append(block(planes, planes, stride=1))
            # self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x): # x is a list of input of length=6 generated by an encoder
        deep_out = []
        for i in range(len(self.layers)):
            inputs = x[-1] if i==0 else out
            out = self.layers[i]([inputs, x[-2-i]])

            if i>=2 and self.use_deep: # deep supervision
                tmp = self.convs[i](out)
                # scale_factor = np.array(self.strides)[i+1:].prod(axis=0).tolist()
                # deep_out += [F.interpolate(tmp, scale_factor=scale_factor, mode='trilinear')]
                deep_out += [F.interpolate(tmp, size=x[0].shape[2:], mode='trilinear')]
                # deep_out += [tmp]
            # if i is the antipenultimate layer
            elif i==(len(self.layers)-3) and self.use_emb: 
                return self.convs[i](out)
        out = self.convs[-1](out)
        if self.use_deep: out = deep_out+[out]
        return out

#---------------------------------------------------------------------------