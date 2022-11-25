
import torch
from torch import nn
import numpy as np

from biom3d.models.head import vgg_mlp
from biom3d.models.encoder_vgg import EncoderBlock, VGGEncoder
from biom3d.models.decoder_vgg_deep import VGGDecoder
from biom3d.models.head import DINOHead, MLPHead

from biom3d.utils import convert_num_pools

#---------------------------------------------------------------------------
# 3D UNet with the previous encoder and decoder

class UNet(nn.Module):
    def __init__(
        self, 
        num_pools=[5,5,5], 
        num_classes=1, 
        factor=32,
        encoder_ckpt = None,
        model_ckpt = None,
        use_deep=True,
        in_planes = 1,
        ):
        super(UNet, self).__init__()
        self.encoder = VGGEncoder(
            EncoderBlock,
            num_pools=num_pools,
            factor=factor,
            in_planes=in_planes,
            )
        self.decoder = VGGDecoder(
            EncoderBlock,
            num_pools=num_pools,
            num_classes=num_classes,
            factor_e=factor,
            factor_d=factor,
            use_deep=use_deep,
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
        
        if model_ckpt is not None:
            print("Load model weights from", model_ckpt)
            if torch.cuda.is_available():
                self.cuda()
            ckpt = torch.load(model_ckpt)
            if 'encoder.last_layer.weight' in ckpt['model'].keys():
                del ckpt['model']['encoder.last_layer.weight']
            print(self.load_state_dict(ckpt['model'], strict=False))

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

    def forward(self, x): 
        # x is an image
        out = self.encoder(x)
        out = self.decoder(out)
        return out

class UNet_old(nn.Module):
    def __init__(self, encoder, decoder):
        super(UNet_old, self).__init__()
        self.encoder = encoder 
        self.decoder = decoder 

    def forward(self, x): # x is an image
        out = self.encoder(x)
        out = self.decoder(out)
        return out

def unet3d_vgg_deep(num_pools=[5,5,5], num_classes=1, factor=32):
    # encoder = ResNetEncoder(EncoderBlock, [1,1,1,1,1], include_top=False)
    # decoder = ResNetDecoder(EncoderBlock, [1,1,1,1,1], num_classes=1)
    encoder = VGGEncoder(EncoderBlock, num_pools=num_pools, factor=factor)
    decoder = VGGDecoder(EncoderBlock, num_pools=num_pools, num_classes=num_classes, factor_e=factor, factor_d=factor)
    unet3d = UNet_old(encoder, decoder)
    return unet3d


def unet3d_vgg_triplet(
    num_pools=[5,5,5], num_classes=1, factor=32,
    encoder_ckpt=None,
    freeze_encoder=True):
    # encoder = ResNetEncoder(EncoderBlock, [1,1,1,1,1], include_top=False)
    # decoder = ResNetDecoder(EncoderBlock, [1,1,1,1,1], num_classes=1)
    encoder = VGGEncoder(EncoderBlock, num_pools=num_pools, factor=factor)

    # load encoder model
    encoder.cuda()
    ckpt = torch.load(encoder_ckpt)
    if 'last_layer.weight' in ckpt['model'].keys():
        del ckpt['model']['last_layer.weight']
    encoder.load_state_dict(ckpt['model'])
    if freeze_encoder:
        for l in encoder.parameters(): 
            print(l.name)
            l.requires_grad = False

    decoder = VGGDecoder(EncoderBlock, num_pools=num_pools, num_classes=num_classes, factor_e=factor, factor_d=factor)
    unet3d = UNet_old(encoder, decoder)
    return unet3d

def unet3d_vgg_triplet_old(
    num_blocks_encoder=[1,1,1,1,1,1], 
    num_blocks_decoder=[1,1,1,1,1,1],
    num_classes=1,
    encoder_ckpt=None,
    freeze_encoder=True):
    # encoder = ResNetEncoder(EncoderBlock, [1,1,1,1,1], include_top=False)
    # decoder = ResNetDecoder(EncoderBlock, [1,1,1,1,1], num_classes=1)
    encoder = VGGEncoder(EncoderBlock, num_blocks_encoder)

    # load encoder model
    encoder.cuda()
    ckpt = torch.load(encoder_ckpt)
    if 'last_layer.weight' in ckpt['model'].keys():
        del ckpt['model']['last_layer.weight']
    encoder.load_state_dict(ckpt['model'])
    if freeze_encoder:
        for l in encoder.parameters(): 
            print(l.name)
            l.requires_grad = False

    decoder = VGGDecoder(EncoderBlock, num_blocks_decoder, num_classes=num_classes)
    unet3d = UNet(encoder, decoder)
    return unet3d

#---------------------------------------------------------------------------
# Unet for cotraining (self-supervised + supervised simultaneously)

# class UNetCoTrain(nn.Module):
#     def __init__(self, encoder, decoder):
#         super(UNetCoTrain, self).__init__()
#         self.encoder = encoder 
#         self.decoder = decoder 

#     def forward(self, x): 
#         # x is a batch of images
#         # the first half of the images are for the unet training
#         # the second half for the encoder
#         # if self.training: print("train")
#         # else: print("test")
#         bs = x.shape[0]//2
#         enc_out = self.encoder(x)
#         dec_in = [enc_out[i][:bs] for i in range(6)] if self.training else enc_out[:6]
#         dec_out = self.decoder(dec_in)
#         if self.training: out = enc_out[-1][bs:], dec_out
#         else: out = dec_out
#         return out

# def unet_cotrain():
#     num_blocks_encoder=[1,1,1,1,1,1]
#     num_blocks_decoder=[1,1,1,1,1,1]
#     num_classes=1
#     encoder = VGGEncoder(EncoderBlock, num_blocks_encoder, include_last=True)
#     decoder = VGGDecoder(EncoderBlock, num_blocks_decoder, num_classes=num_classes)
#     model = UNetCoTrain(encoder, decoder).cuda()
#     return model 


class CotUNet(UNet):
    def __init__(
        self, 
        patch_size,
        num_pools=[5,5,5], 
        num_classes=1, 
        factor=32,
        encoder_ckpt = None,
        in_planes = 1,
        out_dim=2048, # should be equal to the size of the dataset when using arcface or crossentropy loss
        ):

        super().__init__(
            num_pools=num_pools,
            num_classes=num_classes,
            factor=factor,
            model_ckpt=encoder_ckpt,
            use_deep=False,
            in_planes=in_planes,
            )

        # defines head
        strides = np.array(convert_num_pools(num_pools=num_pools))
        strides = (strides[:2]).prod(axis=0)
        in_dim = (np.array(patch_size)/strides).prod().astype(int)*num_classes
        # self.mlp = DINOHead(
        self.mlp = MLPHead(
            in_dim=in_dim,
            out_dim=out_dim,
            nlayers=3,
            bottleneck_dim=256,
            norm_last_layer=True,
            hidden_dim=2048,
        )
    
    def set_num_classes(self, num_classes):
        self.mlp.set_num_classes(num_classes=num_classes)

    def forward(self, x, use_encoder=False, use_last=True): 
        # x is an image
        # use_encoder is a legacy of the NNet model, if yes only the third level of the decoder will be output (instead of the first level)
        out = self.encoder(x)
        self.decoder.use_emb = use_encoder
        out = self.decoder(out)
        if use_encoder:
            out = self.mlp(out, use_last)
            
        return out
        # if use_encoder:
        #     # assert type(out)==list, "[Error] when using 'use_encoder' argument, use_deep must be set to True"
        #     return self.mlp(out[-4])
        # elif type(out)==list:
        #     return out[-1]
        # else:
        #     return out


#---------------------------------------------------------------------------

class NNet(nn.Module):
    def __init__(self, 
        # unet parameters
        num_pools=[5,5,5], 
        num_classes=1, 
        factor=32,
        unet_ckpt = None,
        
        # encoder parameters
        emb_dim=320,
        out_dim=65536,
    ):
        super(NNet, self).__init__()
        self.unet = UNet(
            num_pools=num_pools, 
            num_classes=num_classes, 
            factor=factor,
            encoder_ckpt = None,
            model_ckpt = unet_ckpt,
            use_deep=False,
        )
        self.encoder = vgg_mlp(
            num_pools=num_pools,
            emb_dim=emb_dim,
            out_dim=out_dim,
            factor=factor,
            in_planes=num_classes,
        )
    
    def forward(self, x, use_encoder=False):
        """
        two possible modes here: use_encoder=False, then it is a standard unet model
        use_encoder=False, then the UNet model is followed by a encoder (VGG + Classifier head)
        """
        out = self.unet(x)
        if use_encoder:
            out = self.encoder(out)
        return out 

#---------------------------------------------------------------------------
# UNet cotrain mode: triplet+segmentation

#---------------------------------------------------------------------------