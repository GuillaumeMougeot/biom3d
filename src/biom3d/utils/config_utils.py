import importlib.util
import sys
import shutil
import fileinput
import numpy as np
import os
import yaml # pip install pyyaml
from datetime import datetime

# ----------------------------------------------------------------------------
# config utils
# Convenience class that behaves exactly like dict(), but allows accessing
# the keys and values using the attribute syntax, i.e., "mydict.key = value".
# Author: Terro Keras (progressive_growing_of_gans)

class Dict(dict):
    def __init__(self, *args, **kwargs): super().__init__(*args, **kwargs)
    def __getattr__(self, name): return self[name]
    def __setattr__(self, name, value): self[name] = value
    def __delattr__(self, name): del self[name]

def config_to_type(cfg, new_type):
    """Change config type to a new type. This function is recursive and can be use to change the type of nested dictionaries. 
    """
    old_type = type(cfg)
    cfg = new_type(cfg)
    for k,i in cfg.items():
        if type(i)==old_type:
            cfg[k] = config_to_type(cfg[k], new_type)
    return cfg

def save_yaml_config(path, cfg):
    """
    save a configuration in a yaml file.
    path must thus contains a yaml extension.
    example: path='logs/test.yaml'
    """
    cfg = config_to_type(cfg, dict)
    with open(path, "w") as f:
        yaml.dump(cfg, f, sort_keys=False)
    
def load_yaml_config(path):
    """
    load a yaml stored with the self.save method.
    """
    return config_to_type(compat_old_config(yaml.load(open(path),Loader=yaml.FullLoader)), Dict)

def nested_dict_pairs_iterator(dic):
    ''' This function accepts a nested dictionary as argument
        and iterate over all values of nested dictionaries
        get from: https://thispointer.com/python-how-to-iterate-over-nested-dictionary-dict-of-dicts/ 
    '''
    # Iterate over all key-value pairs of dict argument
    for key, value in dic.items():
        # Check if value is of dict type
        if isinstance(value, dict) or isinstance(value, Dict):
            # If value is dict then iterate over all its values
            for pair in  nested_dict_pairs_iterator(value):
                yield [key, *pair]
        else:
            # If value is not dict type then yield the value
            yield [key, value]

def nested_dict_change_value(dic, key, value):
    """
    Change all value with a given key from a nested dictionary.
    """
    # Loop through all key-value pairs of a nested dictionary and change the value 
    for pairs in nested_dict_pairs_iterator(dic):
        if key in pairs:
            save = dic[pairs[0]]; i=1
            while i < len(pairs) and pairs[i]!=key:
                save = save[pairs[i]]; i+=1
            save[key] = value
    return dic

def replace_line_single(line, key, value):
    """Given a line, replace the value if the key is in the line. This function follows the following format:
    \'key = value\'. The line must follow this format and the output will respect this format. 
    
    Parameters
    ----------
    line : str
        The input line that follows the format: \'key = value\'.
    key : str
        The key to look for in the line.
    value : str
        The new value that will replace the previous one.
    
    Returns
    -------
    line : str
        The modified line.
    
    Examples
    --------
    >>> line = "IMG_PATH = None"
    >>> key = "IMG_PATH"
    >>> value = "path/img"
    >>> replace_line_single(line, key, value)
    IMG_PATH = 'path/img'
    """
    if key==line[:len(key)]:
        assert line[len(key):len(key)+3]==" = ", "[Error] Invalid line. A valid line must contains \' = \'. Line:"+line
        line = line[:len(key)]
        
        # if value is string then we add brackets
        line += " = "
        if isinstance(value,str): 
            line += "\'" + value + "\'"
        elif isinstance(value,np.ndarray):
            line += str(value.tolist())
        else:
            line += str(value)
        line += "\n"
    return line

def replace_line_multiple(line, dic):
    """Similar to replace_line_single but with a dictionary of keys and values.
    """
    for key, value in dic.items():
        line = replace_line_single(line, key, value)
    return line

def save_python_config(
    config_dir,
    base_config = None,
    **kwargs,
    ):
    """
    Save the configuration in a config file. If the path to a base configuration is provided, then update this file with the new auto-configured parameters else use biom3d.config_default file.

    Parameters
    ----------
    config_dir : str
        Path to the configuration folder. If the folder does not exist, then create it.
    base_config : str, default=None
        Path to an existing configuration file which will be updated with the auto-config values.
    **kwargs
        Keyword arguments of the configuration file.

    Returns
    -------
    config_path : str
        Path to the new configuration file.
    
    Examples
    --------
    >>> config_path = save_config_python(\\
        config_dir="configs/",\\
        base_config="configs/pancreas_unet.py",\\
        IMG_PATH="/pancreas/imagesTs_tiny_out",\\
        MSK_PATH="pancreas/labelsTs_tiny_out",\\
        NUM_CLASSES=2,\\
        BATCH_SIZE=2,\\
        AUG_PATCH_SIZE=[56, 288, 288],\\
        PATCH_SIZE=[40, 224, 224],\\
        NUM_POOLS=[3, 5, 5])
    """

    # create the config dir if needed
    if not os.path.exists(config_dir):
        os.makedirs(config_dir, exist_ok=True)

    # name config path with the current date 
    current_time = datetime.now().strftime("%Y%m%d-%H%M%S")

    # copy default config file or use the one given by the user
    if base_config == None:
        try:
            from biom3d import config_default
            config_path = shutil.copy(config_default.__file__, config_dir) 
        except:
            print("[Error] Please provide a base config file or install biom3d.")
            raise RuntimeError
    else: 
        config_path = base_config # WARNING: overwriting!

    # if DESC is in kwargs, then it will be used to rename the config file
    basename = os.path.basename(config_path) if "DESC" not in kwargs.keys() else kwargs['DESC']+'.py'
    new_config_name = os.path.join(config_dir, current_time+"-"+basename)
    os.rename(config_path, new_config_name)

    if base_config is not None:
        # keep a copy of the old file
        shutil.copy(new_config_name, config_path)

    # edit the new config file with the auto-config values
    with fileinput.input(files=(new_config_name), inplace=True) as f:
        for line in f:
            # edit the line
            line = replace_line_multiple(line, kwargs)
            # write back in the input file
            print(line, end='') 
    return new_config_name

def load_python_config(config_path):
    """Return the configuration dictionary given the path of the configuration file.
    The configuration file is in Python format.
    
    Adapted from: https://stackoverflow.com/questions/67631/how-can-i-import-a-module-dynamically-given-the-full-path 
    
    Parameters
    ----------
    config_path : str
        Path of the configuration file. Should have the '.py' extension.
    
    Returns
    -------
    cfg : biom3d.utils.Dict
        Dictionary of the config.
    """
    spec = importlib.util.spec_from_file_location("config", config_path)
    config = importlib.util.module_from_spec(spec)
    sys.modules["config"] = config
    spec.loader.exec_module(config)
    config.CONFIG=compat_old_config(config.CONFIG)
    return config_to_type(config.CONFIG, Dict) # change type from config.Dict to Dict

def recursive_rename_key(d, old_key, new_key):
    if isinstance(d, dict):
        new_dict = {}
        for k, v in d.items():
            k = new_key if k == old_key else k
            new_dict[k] = recursive_rename_key(v, old_key, new_key)
        return new_dict
    elif isinstance(d, list):
        return [recursive_rename_key(i, old_key, new_key) for i in d]
    else:
        return d
    
def compat_old_config(config):
    # Retrocompatibility with old configs
    for k,v in {"IMG_DIR":"IMG_PATH", 
              "MSK_DIR":"MSK_PATH",
              "FG_DIR":"FG_PATH",
              "img_dir":"img_path",
              "msk_dir":"msk_path",
              "fg_dir":'fg_path',
              }.items() :
        config = recursive_rename_key(config,k,v)
    return config

def adaptive_load_config(config_path):
    """Return the configuration dictionary given the path of the configuration file.
    The configuration file is in Python or YAML format.

    Parameters
    ----------
    config_path : str
        Path of the configuration file. Should have the '.py' or '.yaml' extension.
    
    Returns
    -------
    cfg : biom3d.utils.Dict
        Dictionary of the config.
    """
    extension = config_path[config_path.rfind('.'):]
    if extension=='.py':
        config = load_python_config(config_path=config_path)
    elif extension=='.yaml':
        config =  load_yaml_config(config_path)
    else:
        print("[Error] Unknow format for config file.")
    return config