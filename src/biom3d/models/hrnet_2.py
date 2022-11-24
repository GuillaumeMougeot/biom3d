# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Ke Sun (sunk@mail.ustc.edu.cn), Jingyi Xie (hsfzxjy@gmail.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging

import numpy as np

import torch
import torch.nn as nn
import torch._utils
import torch.nn.functional as F

#---------------------------------------------------------------------------
# utils

from biom3d.models.encoder_efficientnet3d import MBConvBlock3D, get_model_params, round_filters, round_repeats

class _MBConvBlock3D(MBConvBlock3D):
    expansion = 1
    block_index = 2

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        _blocks_args, self._global_params = get_model_params(model_name='efficientnet-b4', override_params=None)
        block_args = _blocks_args[self.block_index]

        block_args = block_args._replace(
                # input_filters=round_filters(inplanes, self._global_params),
                # output_filters=round_filters(planes, self._global_params),
                input_filters = inplanes,
                output_filters = planes,
                num_repeat=round_repeats(block_args.num_repeat, self._global_params),
                stride=stride,
        )

        super().__init__(block_args, self._global_params)

def convert_num_pools(num_pools):
    """
    Set adaptive number of pools
        for example: convert [3,5,5] into [[1 2 2],[2 2 2],[2 2 2],[2 2 2],[1 2 2]]
    """
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
    # kernels = (strides*3//2).tolist()
    strides = strides.tolist()
    return strides


class EncoderBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1, option='B', is_last=False):
        super(EncoderBlock, self).__init__()

        self.conv1 = nn.Conv3d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.InstanceNorm3d(planes, affine=True)
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)

        self.is_last = is_last
        if not is_last:
            self.bn2 = nn.InstanceNorm3d(planes, affine=True)

    def forward(self, x):
        out = F.leaky_relu(self.bn1(self.conv1(x)), inplace=True)
        if not self.is_last:
            out = F.leaky_relu(self.bn2(self.conv2(out)), inplace=True)
        else:
            out = self.conv2(out)
        return out


class SmallEncoderBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1, option='B', is_last=False):
        super(SmallEncoderBlock, self).__init__()
        self.conv1 = nn.Conv3d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.is_last = is_last
        if not is_last:
            self.bn1 = nn.InstanceNorm3d(planes, affine=True)

    def forward(self, x):
        if not self.is_last:
            out = F.leaky_relu(self.bn1(self.conv1(x)), inplace=True)
        else:
            out = self.conv1(x)
        return out

#---------------------------------------------------------------------------

ALIGN_CORNERS = True
BN_MOMENTUM = 0.1
logger = logging.getLogger(__name__)
relu_inplace = True # for torch version > 1.10.0

class ModuleHelper:

    @staticmethod
    def BNLeakyReLU(num_features, bn_type=None, **kwargs):
        return nn.Sequential(
            nn.InstanceNorm3d(num_features, **kwargs),
            nn.LeakyReLU()
        )

    @staticmethod
    def BatchNorm2d(*args, **kwargs):
        return nn.InstanceNorm3d


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv3d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class SpatialGather_Module(nn.Module):
    """
        Aggregate the context features according to the initial 
        predicted probability distribution.
        Employ the soft-weighted method to aggregate the context.
    """
    def __init__(self, cls_num=0, scale=1):
        super(SpatialGather_Module, self).__init__()
        self.cls_num = cls_num
        self.scale = scale

    def forward(self, feats, probs):
        batch_size, c, h, w, d = probs.size(0), probs.size(1), probs.size(2), probs.size(3), probs.size(4)
        probs = probs.view(batch_size, c, -1)
        feats = feats.view(batch_size, feats.size(1), -1)
        feats = feats.permute(0, 2, 1) # batch x hwd x c 
        probs = F.softmax(self.scale * probs, dim=2)# batch x k x hwd
        ocr_context = torch.matmul(probs, feats)\
        .permute(0, 2, 1).unsqueeze(3).unsqueeze(4) # batch x k x c
        return ocr_context


class _ObjectAttentionBlock(nn.Module):
    '''
    The basic implementation for object context block
    Input:
        N X C X H X W X D
    Parameters:
        in_channels       : the dimension of the input feature map
        key_channels      : the dimension after the key/query transform
        scale             : choose the scale to downsample the input feature maps (save memory cost)
        bn_type           : specify the bn type
    Return:
        N X C X H X W X D
    '''
    def __init__(self, 
                 in_channels, 
                 key_channels, 
                 scale=1, 
                 bn_type=None):
        super(_ObjectAttentionBlock, self).__init__()
        self.scale = scale
        self.in_channels = in_channels
        self.key_channels = key_channels
        self.pool = nn.MaxPool3d(kernel_size=(scale, scale, scale))
        self.f_pixel = nn.Sequential(
            nn.Conv3d(in_channels=self.in_channels, out_channels=self.key_channels,
                kernel_size=1, stride=1, padding=0, bias=False),
            ModuleHelper.BNLeakyReLU(self.key_channels, bn_type=bn_type),
            nn.Conv3d(in_channels=self.key_channels, out_channels=self.key_channels,
                kernel_size=1, stride=1, padding=0, bias=False),
            ModuleHelper.BNLeakyReLU(self.key_channels, bn_type=bn_type),
        )
        self.f_object = nn.Sequential(
            nn.Conv3d(in_channels=self.in_channels, out_channels=self.key_channels,
                kernel_size=1, stride=1, padding=0, bias=False),
            nn.LeakyReLU(),
            nn.Conv3d(in_channels=self.key_channels, out_channels=self.key_channels,
                kernel_size=1, stride=1, padding=0, bias=False),
            nn.LeakyReLU(),
        )
        self.f_down = nn.Sequential(
            nn.Conv3d(in_channels=self.in_channels, out_channels=self.key_channels,
                kernel_size=1, stride=1, padding=0, bias=False),
            nn.LeakyReLU(),
        )
        self.f_up = nn.Sequential(
            nn.Conv3d(in_channels=self.key_channels, out_channels=self.in_channels,
                kernel_size=1, stride=1, padding=0, bias=False),
            ModuleHelper.BNLeakyReLU(self.in_channels, bn_type=bn_type),
        )

    def forward(self, x, proxy):
        batch_size, h, w, d= x.size(0), x.size(2), x.size(3), x.size(4)
        if self.scale > 1:
            x = self.pool(x)

        query = self.f_pixel(x).view(batch_size, self.key_channels, -1)
        query = query.permute(0, 2, 1)
        key = self.f_object(proxy).view(batch_size, self.key_channels, -1)
        value = self.f_down(proxy).view(batch_size, self.key_channels, -1)
        value = value.permute(0, 2, 1)

        sim_map = torch.matmul(query, key)
        sim_map = (self.key_channels**-.5) * sim_map
        sim_map = F.softmax(sim_map, dim=-1)   

        # add bg context ...
        context = torch.matmul(sim_map, value)
        context = context.permute(0, 2, 1).contiguous()
        context = context.view(batch_size, self.key_channels, *x.size()[2:])
        context = self.f_up(context)
        if self.scale > 1:
            context = F.interpolate(input=context, size=(h, w, d), mode='trilinear', align_corners=ALIGN_CORNERS)

        return context


class ObjectAttentionBlock3D(_ObjectAttentionBlock):
    def __init__(self, 
                 in_channels, 
                 key_channels, 
                 scale=1, 
                 bn_type=None):
        super(ObjectAttentionBlock3D, self).__init__(in_channels,
                                                     key_channels,
                                                     scale, 
                                                     bn_type=bn_type)


class SpatialOCR_Module(nn.Module):
    """
    Implementation of the OCR module:
    We aggregate the global object representation to update the representation for each pixel.
    """
    def __init__(self, 
                 in_channels, 
                 key_channels, 
                 out_channels, 
                 scale=1, 
                 dropout=0.1, 
                 bn_type=None):
        super(SpatialOCR_Module, self).__init__()
        self.object_context_block = ObjectAttentionBlock3D(in_channels, 
                                                           key_channels, 
                                                           scale, 
                                                           bn_type)
        _in_channels = 2 * in_channels

        self.conv_bn_dropout = nn.Sequential(
            nn.Conv3d(_in_channels, out_channels, kernel_size=1, padding=0, bias=False),
            ModuleHelper.BNLeakyReLU(out_channels, bn_type=bn_type),
            nn.Dropout3d(dropout)
        )

    def forward(self, feats, proxy_feats):
        context = self.object_context_block(feats, proxy_feats)

        output = self.conv_bn_dropout(torch.cat([context, feats], 1))

        return output


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.InstanceNorm3d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.LeakyReLU(inplace=relu_inplace)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.InstanceNorm3d(planes, momentum=BN_MOMENTUM)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = out + residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.InstanceNorm3d(planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.InstanceNorm3d(planes, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv3d(planes, planes * self.expansion, kernel_size=1,
                               bias=False)
        self.bn3 = nn.InstanceNorm3d(planes * self.expansion,
                               momentum=BN_MOMENTUM)
        self.relu = nn.LeakyReLU(inplace=relu_inplace)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = out + residual
        out = self.relu(out)

        return out


class HighResolutionModule(nn.Module):
    def __init__(self, num_branches, blocks, num_blocks, num_inchannels,
                 num_channels, fuse_method, multi_scale_output=True, fuse_stride=2):
        super(HighResolutionModule, self).__init__()
        self._check_branches(
            num_branches, blocks, num_blocks, num_inchannels, num_channels)

        self.num_inchannels = num_inchannels
        self.fuse_method = fuse_method
        self.fuse_stride = fuse_stride
        self.num_branches = num_branches

        self.multi_scale_output = multi_scale_output

        self.branches = self._make_branches(
            num_branches, blocks, num_blocks, num_channels,)
        self.fuse_layers = self._make_fuse_layers()
        self.relu = nn.LeakyReLU(inplace=relu_inplace)

    def _check_branches(self, num_branches, blocks, num_blocks,
                        num_inchannels, num_channels):
        if num_branches != len(num_blocks):
            error_msg = 'NUM_BRANCHES({}) <> NUM_BLOCKS({})'.format(
                num_branches, len(num_blocks))
            logger.error(error_msg)
            raise ValueError(error_msg)

        if num_branches != len(num_channels):
            error_msg = 'NUM_BRANCHES({}) <> NUM_CHANNELS({})'.format(
                num_branches, len(num_channels))
            logger.error(error_msg)
            raise ValueError(error_msg)

        if num_branches != len(num_inchannels):
            error_msg = 'NUM_BRANCHES({}) <> NUM_INCHANNELS({})'.format(
                num_branches, len(num_inchannels))
            logger.error(error_msg)
            raise ValueError(error_msg)

    def _make_one_branch(self, branch_index, block, num_blocks, num_channels,
                         stride=1):
        downsample = None
        if stride != 1 or \
           self.num_inchannels[branch_index] != num_channels[branch_index] * block.expansion:
            downsample = nn.Sequential(
                nn.Conv3d(self.num_inchannels[branch_index],
                          num_channels[branch_index] * block.expansion,
                          kernel_size=1, stride=self.fuse_stride[branch_index], bias=False),
                nn.InstanceNorm3d(num_channels[branch_index] * block.expansion,
                            momentum=BN_MOMENTUM),
            )
            stride = self.fuse_stride[branch_index]

        layers = []
        layers.append(block(self.num_inchannels[branch_index],
                            num_channels[branch_index], stride, downsample))
        self.num_inchannels[branch_index] = \
            num_channels[branch_index] * block.expansion
        for i in range(1, num_blocks[branch_index]):
            layers.append(block(self.num_inchannels[branch_index],
                                num_channels[branch_index]))

        return nn.Sequential(*layers)

    def _make_branches(self, num_branches, block, num_blocks, num_channels):
        branches = []

        for i in range(num_branches):
            branches.append(
                self._make_one_branch(i, block, num_blocks, num_channels))

        return nn.ModuleList(branches)

    def _make_fuse_layers(self):
        if self.num_branches == 1:
            return None

        num_branches = self.num_branches
        num_inchannels = self.num_inchannels
        fuse_stride = self.fuse_stride
        fuse_layers = []
        # i index represents the output (that can be multiscale)
        for i in range(num_branches if self.multi_scale_output else 1):
            fuse_layer = []
            # j index represents the input branches
            for j in range(num_branches):
                if j > i:
                    fuse_layer.append(nn.Sequential(
                        nn.Conv3d(num_inchannels[j],
                                  num_inchannels[i],
                                  1,
                                  1,
                                  0,
                                  bias=False),
                        nn.InstanceNorm3d(num_inchannels[i], momentum=BN_MOMENTUM)))
                elif j == i:
                    fuse_layer.append(None)
                else:
                    conv3x3s = []
                    for k in range(i-j):
                        if k == i - j - 1: # last rep
                            num_outchannels_conv3x3 = num_inchannels[i]
                            conv3x3s.append(nn.Sequential(
                                nn.Conv3d(num_inchannels[j],
                                          num_outchannels_conv3x3,
                                          3, fuse_stride[k+j+1], 1, bias=False),
                                nn.InstanceNorm3d(num_outchannels_conv3x3,
                                            momentum=BN_MOMENTUM)))
                        else:
                            num_outchannels_conv3x3 = num_inchannels[j]
                            conv3x3s.append(nn.Sequential(
                                nn.Conv3d(num_inchannels[j],
                                          num_outchannels_conv3x3,
                                          3, fuse_stride[k+j+1], 1, bias=False),
                                nn.InstanceNorm3d(num_outchannels_conv3x3,
                                            momentum=BN_MOMENTUM),
                                nn.LeakyReLU(inplace=relu_inplace)))
                    fuse_layer.append(nn.Sequential(*conv3x3s))
            fuse_layers.append(nn.ModuleList(fuse_layer))

        return nn.ModuleList(fuse_layers)

    def get_num_inchannels(self):
        return self.num_inchannels

    def forward(self, x):
        if self.num_branches == 1:
            return [self.branches[0](x[0])]

        for i in range(self.num_branches):
            x[i] = self.branches[i](x[i])

        x_fuse = []
        for i in range(len(self.fuse_layers)):
            y = x[0] if i == 0 else self.fuse_layers[i][0](x[0])
            for j in range(1, self.num_branches):
                if i == j:
                    y = y + x[j]
                elif j > i:
                    depth_output = x[i].shape[-1]
                    width_output = x[i].shape[-2]
                    height_output = x[i].shape[-3]
                    # y = y + F.interpolate(
                    #     self.fuse_layers[i][j](x[j]),
                    #     size=[height_output, width_output],
                    #     mode='bilinear', align_corners=ALIGN_CORNERS)
                    y = y + F.interpolate(
                        self.fuse_layers[i][j](x[j]),
                        # size=[height_output, width_output, depth_output],
                        size=[height_output, width_output, depth_output],
                        mode='trilinear', align_corners=ALIGN_CORNERS)
                else:
                    y = y + self.fuse_layers[i][j](x[j])
                    
            x_fuse.append(self.relu(y))

        return x_fuse


blocks_dict = {
    'BASIC': BasicBlock,
    'BOTTLENECK': Bottleneck,
    'MB': _MBConvBlock3D,
}


class HighResolutionNet(nn.Module):

    def __init__(self, 
        patch_size,
        num_pools=[5,5,5], 
        num_classes=1, 
        factor=32,
        in_planes = 1,
        feats_only = False,
        encoder_ckpt = None,
        # model_ckpt = None,
        # use_deep=True,
    ):
        global ALIGN_CORNERS
        # extra = config.MODEL.EXTRA
        extra = { #hrnet-18
            'STAGE1': {
                'NUM_MODULES': 1,
                'NUM_BRANCHES': 1,
                'NUM_BLOCKS': [2],
                'NUM_CHANNELS': [64],
                'BLOCK': 'BOTTLENECK',
                'FUSE_METHOD': 'SUM',
            },
            'STAGE2': {
                'NUM_MODULES': 1,
                'NUM_BRANCHES': 2,
                'NUM_BLOCKS': [2,2],
                'NUM_CHANNELS': [18,36],
                'BLOCK': 'BASIC',
                'FUSE_METHOD': 'SUM',
            },
            'STAGE3': {
                'NUM_MODULES': 4,
                'NUM_BRANCHES': 3,
                'NUM_BLOCKS': [2,2,2],
                'NUM_CHANNELS': [18,36,72],
                'BLOCK': 'BASIC',
                'FUSE_METHOD': 'SUM',
            },
            'STAGE4': {
                'NUM_MODULES': 3,
                'NUM_BRANCHES': 4,
                'NUM_BLOCKS': [2,2,2,2],
                'NUM_CHANNELS': [18,36,72,144],
                'BLOCK': 'BASIC',
                'FUSE_METHOD': 'SUM',
            },
        }
        super(HighResolutionNet, self).__init__()
        # ALIGN_CORNERS = config.MODEL.ALIGN_CORNERS

        self.patch_size = patch_size

        strides = convert_num_pools(num_pools)

        # stem net
        # self.stem0 = SmallEncoderBlock(in_planes, factor, stride=1)
        # self.stem1 = SmallEncoderBlock(factor, factor*2, strides[0])
        # self.stem2 = SmallEncoderBlock(factor*2, factor*4, strides[1])

        # self.stem1 = SmallEncoderBlock(in_planes, factor, strides[0])
        # self.stem2 = SmallEncoderBlock(factor, factor*2, strides[1])
        self.conv1 = nn.Conv3d(in_planes, factor*2, kernel_size=3, stride=strides[0], padding=1,
                               bias=False)
        self.bn1 = nn.InstanceNorm3d(factor*2, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv3d(factor*2, factor*2, kernel_size=3, stride=strides[1], padding=1,
                               bias=False)
        self.bn2 = nn.InstanceNorm3d(factor*2, momentum=BN_MOMENTUM)
        self.relu = nn.LeakyReLU(inplace=relu_inplace)

        self.stage1_cfg = extra['STAGE1']
        num_channels = self.stage1_cfg['NUM_CHANNELS'][0]
        block = blocks_dict[self.stage1_cfg['BLOCK']]
        num_blocks = self.stage1_cfg['NUM_BLOCKS'][0]
        self.layer1 = self._make_layer(block, factor*2, num_channels, num_blocks)
        stage1_out_channel = block.expansion*num_channels

        self.stage2_cfg = extra['STAGE2']
        num_channels = self.stage2_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage2_cfg['BLOCK']]
        block.block_index = 3
        # num_channels = [
        #     num_channels[i] * block.expansion for i in range(len(num_channels))]
        self.transition1 = self._make_transition_layer(
            [stage1_out_channel], num_channels, stride=strides[2])
        self.stage2, pre_stage_channels = self._make_stage(
            self.stage2_cfg, num_channels, fuse_stride=strides[1:3])

        self.stage3_cfg = extra['STAGE3']
        num_channels = self.stage3_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage3_cfg['BLOCK']]
        block.block_index = 4
        # num_channels = [
        #     num_channels[i] * block.expansion for i in range(len(num_channels))]
        self.transition2 = self._make_transition_layer(
            pre_stage_channels, num_channels, stride=strides[3])
        self.stage3, pre_stage_channels = self._make_stage(
            self.stage3_cfg, num_channels, fuse_stride=strides[1:4])

        self.stage4_cfg = extra['STAGE4']
        num_channels = self.stage4_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage4_cfg['BLOCK']]
        block.block_index = 5
        # num_channels = [
        #     num_channels[i] * block.expansion for i in range(len(num_channels))]
        self.transition3 = self._make_transition_layer(
            pre_stage_channels, num_channels, stride=strides[4])
        self.stage4, pre_stage_channels = self._make_stage(
            self.stage4_cfg, num_channels, multi_scale_output=True, fuse_stride=strides[1:5])

        # temp? yes!
        # self.upconv1 = nn.ConvTranspose3d(np.int64(np.sum(pre_stage_channels)), factor*2, kernel_size=strides[1], stride=strides[1], bias=False)
        # self.head1 = SmallEncoderBlock(factor*4, factor*2, stride=1)
        # self.upconv2 = nn.ConvTranspose3d(factor*2, factor, kernel_size=strides[0], stride=strides[0], bias=False)
        # self.head2 = SmallEncoderBlock(factor*2, factor*2, stride=1)

        # last_inp_channels = factor*2
        last_inp_channels = np.int64(np.sum(pre_stage_channels))
        # ocr_mid_channels = config.MODEL.OCR.MID_CHANNELS
        # ocr_key_channels = config.MODEL.OCR.KEY_CHANNELS
        ocr_mid_channels = 512
        ocr_key_channels = 256
        # ocr_mid_channels = 64
        # ocr_key_channels = 32

        self.conv3x3_ocr = nn.Sequential(
            nn.Conv3d(last_inp_channels, ocr_mid_channels,
                      kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm3d(ocr_mid_channels),
            nn.LeakyReLU(inplace=relu_inplace),
        )
        self.ocr_gather_head = SpatialGather_Module(num_classes)

        self.ocr_distri_head = SpatialOCR_Module(in_channels=ocr_mid_channels,
                                                 key_channels=ocr_key_channels,
                                                 out_channels=ocr_mid_channels,
                                                 scale=1,
                                                 dropout=0.05,
                                                 )
        self.cls_head = nn.Conv3d(
            ocr_mid_channels, num_classes, kernel_size=1, stride=1, padding=0, bias=True)
        
        self.aux_head = nn.Sequential(
            nn.Conv3d(last_inp_channels, last_inp_channels,
                      kernel_size=1, stride=1, padding=0),
            nn.InstanceNorm3d(last_inp_channels),
            nn.LeakyReLU(inplace=relu_inplace),
            nn.Conv3d(last_inp_channels, num_classes,
                      kernel_size=1, stride=1, padding=0, bias=True)
        )
        
        # unet head
        # self.upconv1 = nn.ConvTranspose3d(ocr_mid_channels, factor, kernel_size=strides[1], stride=strides[1], bias=False)
        # self.head1 = SmallEncoderBlock(factor*2, factor*2, stride=1)
        # self.upconv2 = nn.ConvTranspose3d(factor*2, factor, kernel_size=strides[0], stride=strides[0], bias=False)
        # self.head2 = SmallEncoderBlock(factor, factor, stride=1)
        # self.head3 = nn.Conv3d(factor, num_classes, kernel_size=1, stride=1, padding=0, bias=False)

        self.feats_only = feats_only # whether to output features only

        # load encoder if needed
        if encoder_ckpt is not None:
            print("Load encoder weights from", encoder_ckpt)
            if torch.cuda.is_available():
                self.cuda()
            ckpt = torch.load(encoder_ckpt)
            # if 'last_layer.weight' in ckpt['model'].keys():
            #     del ckpt['model']['last_layer.weight']
            if 'model' in ckpt.keys():
                # remove `module.` prefix
                state_dict = {k.replace("module.", ""): v for k, v in ckpt['model'].items()} 
                # remove `0.` prefix induced by the sequential wrapper
                state_dict = {k[k.startswith("0.") and len("0."):]: v for k, v in state_dict.items()}  
                print(self.load_state_dict(state_dict, strict=False))
            elif 'teacher' in ckpt.keys():
                # remove `module.` prefix
                state_dict = {k.replace("module.", ""): v for k, v in ckpt['teacher'].items()}  
                # remove `backbone.` prefix induced by multicrop wrapper
                state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
                print(self.load_state_dict(state_dict, strict=False))
            else:
                print("[Warning] the following encoder couldn't be loaded, wrong key:", encoder_ckpt)
        
    def _make_transition_layer(
            self, num_channels_pre_layer, num_channels_cur_layer, stride=2):
        num_branches_cur = len(num_channels_cur_layer)
        num_branches_pre = len(num_channels_pre_layer)

        transition_layers = []
        for i in range(num_branches_cur):
            if i < num_branches_pre:
                if num_channels_cur_layer[i] != num_channels_pre_layer[i]:
                    transition_layers.append(nn.Sequential(
                        nn.Conv3d(num_channels_pre_layer[i],
                                  num_channels_cur_layer[i],
                                  kernel_size=3,
                                  stride=1,
                                  padding=1,
                                  bias=False),
                        nn.InstanceNorm3d(
                            num_channels_cur_layer[i], momentum=BN_MOMENTUM),
                        nn.LeakyReLU(inplace=relu_inplace)))
                else:
                    transition_layers.append(None)
            else:
                conv3x3s = []
                for j in range(i+1-num_branches_pre):
                    inchannels = num_channels_pre_layer[-1]
                    outchannels = num_channels_cur_layer[i] \
                        if j == i-num_branches_pre else inchannels
                    conv3x3s.append(nn.Sequential(
                        nn.Conv3d(
                            inchannels, outchannels, kernel_size=3, stride=stride, padding=1, bias=False),
                        nn.InstanceNorm3d(outchannels, momentum=BN_MOMENTUM),
                        nn.LeakyReLU(inplace=relu_inplace)))
                transition_layers.append(nn.Sequential(*conv3x3s))

        return nn.ModuleList(transition_layers)

    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv3d(inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.InstanceNorm3d(planes * block.expansion, momentum=BN_MOMENTUM),
            )

        layers = []
        layers.append(block(inplanes, planes, stride, downsample))
        inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(inplanes, planes))

        return nn.Sequential(*layers)

    def _make_stage(self, layer_config, num_inchannels,
                    multi_scale_output=True, fuse_stride=2):
        num_modules = layer_config['NUM_MODULES']
        num_branches = layer_config['NUM_BRANCHES']
        num_blocks = layer_config['NUM_BLOCKS']
        num_channels = layer_config['NUM_CHANNELS']
        block = blocks_dict[layer_config['BLOCK']]
        fuse_method = layer_config['FUSE_METHOD']

        modules = []
        for i in range(num_modules):
            # multi_scale_output is only used last module
            if not multi_scale_output and i == num_modules - 1:
                reset_multi_scale_output = False
            else:
                reset_multi_scale_output = True
            modules.append(
                HighResolutionModule(num_branches,
                                     block,
                                     num_blocks,
                                     num_inchannels,
                                     num_channels,
                                     fuse_method,
                                     reset_multi_scale_output,
                                     fuse_stride=fuse_stride)
            )
            num_inchannels = modules[-1].get_num_inchannels()

        return nn.Sequential(*modules), num_inchannels

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        # stem1 = self.stem0(x)
        # stem2 = self.stem1(stem1)
        # stem2 = self.stem1(x)
        # x = self.stem2(stem2)
        
        x = self.layer1(x)

        x_list = []
        for i in range(self.stage2_cfg['NUM_BRANCHES']):
            if self.transition1[i] is not None:
                x_list.append(self.transition1[i](x))
            else:
                x_list.append(x)
        y_list = self.stage2(x_list)

        x_list = []
        for i in range(self.stage3_cfg['NUM_BRANCHES']):
            if self.transition2[i] is not None:
                if i < self.stage2_cfg['NUM_BRANCHES']:
                    x_list.append(self.transition2[i](y_list[i]))
                else:
                    x_list.append(self.transition2[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage3(x_list)

        x_list = []
        for i in range(self.stage4_cfg['NUM_BRANCHES']):
            if self.transition3[i] is not None:
                if i < self.stage3_cfg['NUM_BRANCHES']:
                    x_list.append(self.transition3[i](y_list[i]))
                else:
                    x_list.append(self.transition3[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        x = self.stage4(x_list)

        # Upsampling
        x0_h, x0_w, x0_d = x[0].size(2), x[0].size(3), x[0].size(4)
        x1 = F.interpolate(x[1], size=(x0_h, x0_w, x0_d),
                        mode='trilinear', align_corners=ALIGN_CORNERS)
        x2 = F.interpolate(x[2], size=(x0_h, x0_w, x0_d),
                        mode='trilinear', align_corners=ALIGN_CORNERS)
        x3 = F.interpolate(x[3], size=(x0_h, x0_w, x0_d),
                        mode='trilinear', align_corners=ALIGN_CORNERS)

        feats = torch.cat([x[0], x1, x2, x3], 1) # pixel representations

        if self.feats_only:
            return feats

        # feats = self.upconv1(feats)
        # feats = torch.cat([feats,stem2], dim=1)
        # feats = self.head1(feats)
        
        # feats = self.upconv2(feats)
        # feats = torch.cat([feats,stem1], dim=1)
        # feats = self.head2(feats)

        out_aux_seg = []

        # ocr
        out_aux = self.aux_head(feats) # soft object regions
        # compute contrast feature
        feats = self.conv3x3_ocr(feats)

        context = self.ocr_gather_head(feats, out_aux) # object region representations 
        feats = self.ocr_distri_head(feats, context) # object contextual representations

        
        
        # unet head
        out = self.cls_head(feats) # augmented representations

        # out = self.upconv1(feats)
        # out = torch.cat([out,stem2], dim=1)
        # out = self.head1(out)
        
        # out = self.upconv2(out)
        # out = torch.cat([out,stem1], dim=1)
        # out = self.head2(out)

        # out = self.head3(out)

        if not self.training:
            out = F.interpolate(out, size=self.patch_size,
                        mode='trilinear', align_corners=ALIGN_CORNERS)
            return out
        
        # if self.training:
        out_aux_seg.append(out_aux)
        out_aux_seg.append(out)

        return out_aux_seg
        # else:
        #     out = F.interpolate(out, size=self.patch_size,
        #                 mode='trilinear', align_corners=ALIGN_CORNERS)
        #     return out 

    def init_weights(self, pretrained='',):
        logger.info('=> init weights from normal distribution')
        for name, m in self.named_modules():
            if any(part in name for part in {'cls', 'aux', 'ocr'}):
                # print('skipped', name)
                continue
            if isinstance(m, nn.Conv3d):
                nn.init.normal_(m.weight, std=0.001)
            # elif isinstance(m, BatchNorm2d_class):
            #     nn.init.constant_(m.weight, 1)
            #     nn.init.constant_(m.bias, 0)
        if os.path.isfile(pretrained):
            pretrained_dict = torch.load(pretrained, map_location={'cuda:0': 'cpu'})
            logger.info('=> loading pretrained model {}'.format(pretrained))
            model_dict = self.state_dict()
            pretrained_dict = {k.replace('last_layer', 'aux_head').replace('model.', ''): v for k, v in pretrained_dict.items()}  
            print(set(model_dict) - set(pretrained_dict))            
            print(set(pretrained_dict) - set(model_dict))            
            pretrained_dict = {k: v for k, v in pretrained_dict.items()
                               if k in model_dict.keys()}
            # for k, _ in pretrained_dict.items():
                # logger.info(
                #     '=> loading {} pretrained model {}'.format(k, pretrained))
            model_dict.update(pretrained_dict)
            self.load_state_dict(model_dict)
        elif pretrained:
            raise RuntimeError('No such file {}'.format(pretrained))


# def get_seg_model(cfg, **kwargs):
#     model = HighResolutionNet(cfg, **kwargs)
#     model.init_weights(cfg.MODEL.PRETRAINED)

#     return model