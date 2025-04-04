import argparse
from enum import Enum
import os
from biom3d import utils
import torch

def load_model(path_to_model, best):
    model_dir = os.path.join(path_to_model, 'model')
    model_name = utils.load_yaml_config(os.path.join(path_to_model,"log","config.yaml")).DESC+("_best" if best else "")+'.pth'
    ckpt_path = os.path.join(model_dir, model_name)
    print("Loading model from", ckpt_path)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    return torch.load(ckpt_path,map_location=torch.device(device),weights_only = True)

#Based on https://github.com/bioimage-io/core-bioimage-io-python/blob/53dfc45cf23351da61e8b22d100d77fb54c540e6/example/model_creation.ipynb
def packagev0x5BIZ(path_to_model,output = None,best = False):
    ckpt = load_model(path_to_model,best)
    
class Target(Enum ):    
    v0x5BIZ = "v0.5BioImageZoo"

    def __str__(self):
        return self.value

if __name__=='__main__':
    parser = argparse.ArgumentParser(description="Package model.")
    parser.add_argument("-t", "--target", type=Target, default=Target.v0x5BIZ, choices=list(Target),help="Target image and version")
    parser.add_argument("-o", "--output_dir", type=str, default=None,help="Output directory, will be created if needed (default current directory)")
    parser.add_argument("-b", "--best", action = "store_true",help="Whether best model is used")
    parser.add_argument("model_dir",help="Path to model directory")  
    args = parser.parse_args()
    if(args.target == Target.v0x5BIZ):
        packagev0x5BIZ(args.model_dir,args.output_dir,args.best)