#---------------------------------------------------------------------------
# 3D efficient net stolen from:
# https://github.com/shijianjian/EfficientNet-PyTorch-3D 
#
# usage:
# model = EfficientNet3D.from_name("efficientnet-b1", override_params={'include_top': False}, in_channels=1)
# model.cuda()
#---------------------------------------------------------------------------

import torch
from torch import nn

def get_layer(model, layer_names):
    """
    get a layer from a model from a list of its module and submodules
    e.g.: l = ['_blocks','0','_depthwise_conv']
    """
    for e in layer_names: 
        model = model._modules[e]
    return model

def get_pyramid(model, pyramid):
    """
    return a list of layers from the model described by the dictionary called 'pyramid'.
    e.g.: 
    pyramid = {
        0: ['_conv_stem'],              # 100
        1: ['_blocks', '1', '_bn0'],    # 50
        2: ['_blocks', '3', '_bn0'],    # 25
        3: ['_blocks', '5', '_bn0'],    # 12
        4: ['_blocks', '11', '_bn0'],   # 6
        5: ['_bn1']                     # 3
    }
    """
    layers = []
    for v in pyramid.values():
        layers += [get_layer(model, v)]
    return layers

def get_outfmaps(layer):
    """
    return the depth of output feature map.
    """
    s = layer.weight.shape
    if len(s)==5 or len(s)==1:
        return s[0]
    else:
        print("Error")
        return 0

def conv_block(in_dim, out_dim, activation, stride=1, norm_fn=nn.InstanceNorm3d):
    return nn.Sequential(
        nn.Conv3d(in_dim, out_dim, kernel_size=3, stride=1, padding=1, bias=False),
        norm_fn(out_dim, affine=True),
        activation,)

def conv_block_2(in_dim, out_dim, activation, stride=1):
    return nn.Sequential(
        conv_block(in_dim, out_dim, activation,stride=stride),
        conv_block(out_dim, out_dim, activation,stride=1),)

# def conv_trans_block(in_dim, out_dim, activation=None, norm_fn=nn.InstanceNorm3d):
#     if activation is not None:
#         return nn.Sequential(
#             nn.ConvTranspose3d(in_dim, out_dim, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
#             norm_fn(out_dim, affine=True),
#             activation,)
#     else:
#         return nn.Sequential(
#             nn.ConvTranspose3d(in_dim, out_dim, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
#             norm_fn(out_dim, affine=True),)

def conv_trans_block(in_dim, out_dim):
    return nn.ConvTranspose3d(in_dim, out_dim, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False)


class ResBlock(nn.Module):
    def __init__(self, in_dim, out_dim, activation, stride=1, norm_fn=nn.InstanceNorm3d):
        super(ResBlock, self).__init__()
        self.act = activation 
        self.norm = norm_fn(in_dim) 
        self.convx = nn.Conv3d(in_dim, out_dim, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv1 = conv_block(in_dim, out_dim, activation)
        self.conv2 = nn.Conv3d(out_dim, out_dim, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, x):
        # out = self.act(x)
        x = self.norm(x)
        x = self.act(x)
        out = self.conv1(x)
        out = self.conv2(out)
        out = out + self.convx(x)
        return out 

class FPN(nn.Module):
    def __init__(self,
                 encoder,
                 pyramid,
                 in_dim,
                 out_dim,
                 num_filters,
                 encoder_ckpt=None,
                 prob_dropout=0.1,
                ):
        super(FPN, self).__init__()
        
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_filters = num_filters
        activation = nn.LeakyReLU(0.01, inplace=True)
        
        # down
        self.encoder = encoder
        if encoder_ckpt is not None:
            print("Load encoder weights from", encoder_ckpt)
            if torch.cuda.is_available():
                self.encoder.cuda()
            ckpt = torch.load(encoder_ckpt)
            if 'last_layer.weight' in ckpt['model'].keys():
                del ckpt['model']['last_layer.weight']
            self.encoder.load_state_dict(ckpt['model'])
        self.pyramid = get_pyramid(encoder, pyramid) # only the first five elements of the list are used

        # hook the pyramid
        self.down = {}
        for i in range(len(self.pyramid)):
            self.pyramid[i].register_forward_hook(self.get_activation(i))
        
        # up
        # dim = get_outfmaps(self.pyramid[-1])
        # self.layers = [ResBlock(dim, dim, activation, prob_dropout=prob_dropout)]
        self.layers = []
        self.up = []
        for i in range(len(pyramid)-1):
            low_dim = get_outfmaps(self.pyramid[-1-i])
            high_dim = get_outfmaps(self.pyramid[-2-i])
            self.up += [conv_trans_block(low_dim, high_dim)]
            # self.layers += [conv_block_2(high_dim*2, high_dim, activation)]
            self.layers += [
                nn.Sequential(
                    nn.Dropout(p=prob_dropout),
                    nn.Conv3d(high_dim*2, high_dim, kernel_size=3, stride=1, padding=1, bias=False),
                    ResBlock(high_dim, high_dim, activation),
                    ResBlock(high_dim, high_dim, activation),
                )
            ]

        self.layers = nn.ModuleList(self.layers)
        self.up = nn.ModuleList(self.up)

        # out
        out_fmaps=4
        dim = get_outfmaps(self.pyramid[0])
        self.conv_out = conv_block(in_dim, out_fmaps, activation)
        self.up_out = conv_trans_block(dim, out_fmaps)
        self.out = nn.Sequential(
            ResBlock(out_fmaps*2, out_fmaps, activation),
            nn.Conv3d(out_fmaps, out_dim, kernel_size=1, stride=1, padding=0, bias=False),
        )
    
    def freeze_encoder(self, freeze=True):
        """
        freeze or unfreeze encoder model
        """
        if freeze:
            print("Freezing encoder weights...")
        else:
            print("Unfreezing encoder weights...")
        for l in self.encoder.parameters(): 
            l.requires_grad = freeze
    
    def unfreeze_encoder(self):
        self.freeze_encoder(False)

    def get_activation(self, name):
        def hook(model, input, output):
            self.down[name] = [output]
        return hook
        
    def forward(self, x):
        # down
        self.encoder(x)

        # up
        for i in range(len(self.up)):
            if i==0: low = self.up[0](self.down[len(self.down)-1][0])
            else: low = self.up[i](out)
            out = torch.cat([low, self.down[len(self.down)-2-i][0]], dim=1)
            out = self.layers[i](out)
        
        # output
        x = self.conv_out(x)
        out = self.up_out(out)
        out = torch.cat([x, out], dim=1)
        out = self.out(out)
        return out

if __name__=='__main__':
    from models.encoder_efficientnet3d import EfficientNet3D
    model = EfficientNet3D.from_name("efficientnet-b0", override_params={'include_top': False}, in_channels=32)
    pyramid = {
        0: ['_conv_stem'],              # 100
        1: ['_blocks', '1', '_bn0'],    # 50
        2: ['_blocks', '3', '_bn0'],    # 25
        3: ['_blocks', '5', '_bn0'],    # 12
        4: ['_blocks', '11', '_bn0'],   # 6
        5: ['_bn1']                     # 3
    }

    fpn = FPN(encoder=model,pyramid=pyramid,in_dim=1,out_dim=1,num_filters=32).cuda()
    print(fpn(torch.rand(1,1,128,128,128).cuda()))