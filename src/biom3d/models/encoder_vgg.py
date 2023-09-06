#---------------------------------------------------------------------------
# 3D Resnet adapted from:
# https://github.com/akamaster/pytorch_resnet_cifar10 
#---------------------------------------------------------------------------

import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from biom3d.utils import convert_num_pools

#---------------------------------------------------------------------------
# 3D Resnet encoder

__all__ = ['ResNet', 'resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet110', 'resnet1202']

def _weights_init(m):
    # classname = m.__class__.__name__
    #print(classname)
    # if isinstance(m, nn.Linear) or isinstance(m, nn.Conv3d):
    #     init.kaiming_normal_(m.weight)
    # if isinstance(m, nn.Conv3d):
    #     nn.init.xavier_normal_(m.weight)
    # elif isinstance(m, nn.BatchNorm3d) or isinstance(m, nn.BatchNorm1d):
    #     nn.init.constant_(m.weight, 1)
    #     nn.init.constant_(m.bias, 0)
    # elif isinstance(m, nn.Linear):
    #     nn.init.xavier_normal_(m.weight)
    #     if m.bias is not None:
    #         nn.init.constant_(m.bias, 0)
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
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)

class GlobalAvgPool3d(nn.Module):
    def __init__(self):
        super(GlobalAvgPool3d, self).__init__()

    def forward(self,x):
        out = x.mean(dim=(-3,-2,-1))
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

class VGGEncoder(nn.Module):
    def __init__(self, 
        block, 
        num_pools, 
        factor = 32,
        first_stride=[1,1,1], # the stride of the first layer convolution
        flip_strides = False, # whether to invert strides order. Flipped strides creates larger feature maps.
        use_emb=False, # use the embedding output (along with the existing ones)
        emb_dim=320,
        use_head=False,
        patch_size = None, # only needed when using the head
        in_planes = 1,
        ): 
        super(VGGEncoder, self).__init__()
        factors = [factor * i for i in [1,2,4,8,10,10,10]]
        self.in_planes = in_planes
        self.use_emb=use_emb
        self.use_head = use_head

        # computes the strides
        # for example: convert [3,5,5] into [[1 1 1],[1 2 2],[2 2 2],[2 2 2],[2 2 2],[1 2 2]]
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
        strides = convert_num_pools(num_pools=num_pools)
        if flip_strides: strides = np.flip(strides, axis=0)
        strides = np.vstack(([first_stride],strides))
        strides = strides.tolist()
        

        # computes the strides (old)
        # strides = [first_stride]
        # num_pools_ = np.array(num_pools)
        # while num_pools_.sum()!=0:
        #     stride = (num_pools_>0).astype(int)
        #     num_pools_ -= stride
        #     strides += [list(stride+1)]

        # defines the network
        self.layers = []
        for i in range(max(num_pools)+1):
            self.layers += [self._make_layer(block, factors[i], num_blocks=1, stride=strides[i], is_last=(i==max(num_pools) and use_emb))]
        
        self.layers = nn.ModuleList(self.layers)

        if use_emb:
            # factor = factors[max(num_pools)]
            # emb_size = factor # size of the embedding
            # self.global_pool=GlobalAvgPool3d()
            # self.global_pool=torch.nn.AdaptiveAvgPool3d((1,1,1))
            # self.global_pool=torch.nn.AdaptiveAvgPool3d((5,5,5))
            # self.fc = nn.Linear(factor, emb_dim, bias=True)
            # self.bn = nn.BatchNorm1d(emb_dim)
            # self.bn.weight.requires_grad = False
            if use_head:
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

    def _make_layer(self, block, planes, num_blocks, stride, is_last=False):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers += [block(self.in_planes, planes, stride, is_last=is_last)]
            self.in_planes = planes

        return nn.Sequential(*layers)

    def forward(self, x, use_encoder=False):
        # stores the intermediate outputs
        out = []
        for i in range(len(self.layers)):
            inputs = x if i==0 else out[-1]
            out += [self.layers[i](inputs)]
            
        if self.use_emb:
            # out = self.last_layer(out[-1])
            # out = self.global_pool(out[-1])
            # out = torch.flatten(out, 1)
            out = out[-1].view(out[-1].size(0), -1)
            # out = nn.functional.normalize(out, dim=-1, p=2)
            # if self.use_head:
            #     out = nn.functional.normalize(out, dim=-1, p=2)
            #     out = self.fc(out)
            if use_encoder:
                out = self.head(out)
            
        return out
#---------------------------------------------------------------------------

