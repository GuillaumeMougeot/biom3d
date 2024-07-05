#---------------------------------------------------------------------------
# 3D efficient net stolen from:
# https://github.com/shijianjian/EfficientNet-PyTorch-3D 
#
# usage:
# model = EfficientNet3D.from_name("efficientnet-b1", override_params={'include_top': False}, in_channels=1)
# model.cuda()
#---------------------------------------------------------------------------

from biom3d.models.encoder_vgg import EncoderBlock, SmallEncoderBlock
from biom3d.models.decoder_vgg_deep import VGGDecoder
from biom3d.models.encoder_efficientnet3d import EfficientNet3D, efficientnet3d

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
    def __init__(
        self, 
        patch_size,
        num_pools=[5,5,5], 
        num_classes=1, 
        factor=32,
        encoder_ckpt = None,
        model_ckpt = None,
        use_deep=True,
        in_planes = 1,
        ):
        super(EffUNet, self).__init__()

        pyramid={                       # efficientnet b4
        0: ['_bn0'],                    
        1: ['_blocks', '1', '_bn2'],   
        2: ['_blocks', '5', '_bn2'],  
        3: ['_blocks', '9', '_bn2'],    
        4: ['_blocks', '21', '_bn2'],  
        5: ['_blocks', '31', '_bn2'],  
        }
        # pyramid={                       # efficientnet b2
        # 0: ['_bn0'],                    
        # 1: ['_blocks', '1', '_bn2'],   
        # 2: ['_blocks', '4', '_bn2'],  
        # 3: ['_blocks', '7', '_bn2'],    
        # 4: ['_blocks', '15', '_bn2'],  
        # 5: ['_blocks', '22', '_bn2'],  
        # }
        blocks_args, global_params = efficientnet3d(
            # width_coefficient=1.1, # efficientnet b2
            # depth_coefficient=1.2,
            # dropout_rate=0.3,
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
            ckpt = torch.load(encoder_ckpt)
            # if 'last_layer.weight' in ckpt['model'].keys():
            #     del ckpt['model']['last_layer.weight']
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
            ckpt = torch.load(model_ckpt)
            if 'encoder.last_layer.weight' in ckpt['model'].keys():
                del ckpt['model']['encoder.last_layer.weight']
            self.load_state_dict(ckpt['model'])

    def freeze_encoder(self, freeze=True):
        """
        freeze or unfreeze encoder model
        """
        if freeze:
            print("Freezing encoder weights...")
        else:
            print("Unfreezing encoder weights...")
        for l in self.encoder.parameters(): 
            l.requires_grad = not freeze
    
    def unfreeze_encoder(self):
        self.freeze_encoder(False)

    def get_activation(self, name):
        def hook(model, input, output):
            self.down[name] = output
            # self.down += [output] # CAREFUL MEMORY LEAK! when using list make sure to empty it during the forward pass!
        return hook

    def forward(self, x): # x is an image
        self.encoder(x)
        out = self.decoder(list(self.down.values()))
        # del self.down # CAREFUL MEMORY LEAK! when using list make sure to empty it during the forward pass!
        # self.down = []
        return out

#---------------------------------------------------------------------------
