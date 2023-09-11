#---------------------------------------------------------------------------
# Graphical User Interface for Biom3d
# WARNING: with this current version the remote access mode only work with a
# linux server.
# WARNING: this script is only meant to work as main script and not as a
# module. The imports are not included outside the __main__. 
# Content:
#  * Imports
#  * Constants definition
#  * Tkinter style definition
#  * ProxyJump helper class
#  * File dialog helper class
#  * Preprocess tab
#  * Training tab
#  * Predict tab
#  * [deprecated] Omero tab
#  * Tkinter mainloop
#---------------------------------------------------------------------------

# from tkinter import *
import tkinter as tk
from tkinter import LEFT, ttk, Tk, N, W, E, S, YES, IntVar, StringVar
from tkinter import filedialog
import paramiko
from stat import S_ISDIR, S_ISREG # for recursive download
import os 
import yaml
from sys import platform 
import sys
# if platform=='linux': # only import if linux because windows omero plugin requires Visual Studio Install which is too big

try:
    from biom3d.config_default import CONFIG

    from biom3d.preprocess import auto_config_preprocess
    from biom3d.utils import save_python_config
    from biom3d.utils import adaptive_load_config
    # the packages below are only needed for the local version of the GUI
    # WARNING! the lines below must be commented when deploying the remote version,
    # and uncommented when installing the local version.
    from biom3d.pred import pred
    from biom3d.builder import Builder
    import biom3d.omero_pred
    from biom3d.utils import load_python_config
    from biom3d.train import train
    import biom3d.upload_pred
    import torch
except:
    pass

import numpy as np

#----------------------------------------------------------------------------
# Constants 
# remote or local

REMOTE = False

VENV = "" # virtual environment path

# The option below is made to remove the 'start locally' button in the gui. This is
# useful for the deployment only in order to reduce the size of the 
# distribution we only allow remote access. 
LOCAL = False 

MAIN_DIR = "/home/biome/biom3d" # folder containing biom3d repository on server computer
TRANSPORT = False

# style
PADDING = "3 3 3 3"

FRAME_STYLE = 'white_style.TFrame'
LABELFRAME_STYLE = 'white_style.TLabelframe'
LABELFRAME_LABEL_STYLE = 'white_style.TLabelframe.Label'
BUTTON_STYLE = 'white_style.TButton'
ROOT_FRAME_STYLE = 'red_style.TFrame'
NOTEBOOK_STYLE = 'red_style.TNotebook'
NOTEBOOK_TAB_STYLE = 'red_style.TNotebook.Tab'

#----------------------------------------------------------------------------
# Styles 

def init_styles():

    white = "#FFFFFF"
    red = "#D09696"

    # Main style settings
    theme_style = ttk.Style()
    theme_style_list = ['TFrame', 'TLabel', 'TLabelframe', 'TLabelframe.Label', 'TButton', 'TNotebook', 'TNotebook.Tab', 'TCheckbutton', 'TPanedwindow']
    
    ## Background setup
    for i in range(len(theme_style_list)):
        theme_style.configure(theme_style_list[i],background=white)

    ## Frame setup
    
    ## Notebook tab setup
    theme_style.configure('TNotebook.Tab', padding=[25,5], background=red)
    # expand: enlarge the tab size when selected
    theme_style.map('TNotebook.Tab', background=[("selected", white)], expand=[("selected", [1,1,1,0])])

    red_style = ttk.Style()
    red_style_name = 'red'
    red_style_list = ['TFrame','TNotebook', 'TButton']
    for i in range(len(red_style_list)):
        red_style_list[i] = red_style_name+'.'+red_style_list[i]
        red_style.configure(red_style_list[i], background=red)
    ## Notebook setup
    red_style.configure('red.TNotebook', tabmargins=[1, 0, 1, 0])

#----------------------------------------------------------------------------
# FTP utils

def ftp_put_file(ftp, localpath, remotepath):
    """
    put a file on a ftp client. Assert that the remetopath does not exist otherwise skip the copy.
    """
    try:
        print(ftp.stat(remotepath))
        print('file {} already exist'.format(remotepath))
    except:
        print("copying {} to {}".format(localpath, remotepath))
        ftp.put(localpath=localpath, remotepath=remotepath)

def ftp_put_folder(ftp, localpath, remotepath):
    """
    this function is recursive, if the folder contains subfolder it will call itself until leaf files. 
    """
    global REMOTE # must be defined to create folders 

    # create path to remote if needed
    REMOTE.exec_command("mkdir -p {}".format(remotepath))

    # copy each individual file to remote
    list_files = os.listdir(localpath)
    for i in range(len(list_files)):
        localpath_ = os.path.join(localpath, list_files[i])
        remotepath_ = remotepath+"/"+list_files[i]
        print("local", localpath_)
        print("remote", remotepath_)
        if os.path.isdir(localpath_):
            ftp_put_folder(ftp, localpath_, remotepath_)
        else:
            ftp_put_file(ftp, localpath_, remotepath_)

# from https://stackoverflow.com/questions/6674862/recursive-directory-download-with-paramiko
def ftp_get_folder(ftp, remotedir, localdir):
    """
    download folder through ftp 
    from https://stackoverflow.com/questions/6674862/recursive-directory-download-with-paramiko 
    """
    for entry in ftp.listdir_attr(remotedir):
        remotepath = remotedir + "/" + entry.filename
        localpath = os.path.join(localdir, entry.filename)
        mode = entry.st_mode
        if S_ISDIR(mode):
            try:
                os.mkdir(localpath)
            except OSError:     
                pass
            ftp_get_folder(ftp, remotepath, localpath)
        elif S_ISREG(mode):
            ftp.get(remotepath, localpath)

#----------------------------------------------------------------------------
# ProxyJump
# from https://stackoverflow.com/questions/42208655/paramiko-nest-ssh-session-to-another-machine-while-preserving-paramiko-function

import time
import socket     
from select import select                                                       


class ParaProxy(paramiko.proxy.ProxyCommand):                      
    def __init__(self, stdin, stdout, stderr):                             
        self.stdin = stdin                                                 
        self.stdout = stdout                                               
        self.stderr = stderr
        self.timeout = None
        self.channel = stdin.channel                                               

    def send(self, content):                                               
        try:                                                               
            self.stdin.write(content)                                      
        except IOError as exc:                                             
            raise socket.error("Error: {}".format(exc))                                                    
        return len(content)                                                

    def recv(self, size):                                                  
        try:
            buffer = b''
            start = time.time()

            while len(buffer) < size:
                select_timeout = self._calculate_remaining_time(start)
                ready, _, _ = select([self.stdout.channel], [], [],
                                     select_timeout)
                if ready and self.stdout.channel is ready[0]:
                      buffer += self.stdout.read(size - len(buffer))

        except socket.timeout:
            if not buffer:
                raise

        except IOError as e:
            return ""

        return buffer

    def _calculate_remaining_time(self, start):
        if self.timeout is not None:
            elapsed = time.time() - start
            if elapsed >= self.timeout:
                raise socket.timeout()
            return self.timeout - elapsed
        return None                                   

    def close(self):                                                       
        self.stdin.close()                                                 
        self.stdout.close()                                                
        self.stderr.close()
        self.channel.close()        

#----------------------------------------------------------------------------
# general utils

class Dict(dict):
    def __init__(self, *args, **kwargs): super().__init__(*args, **kwargs)
    def __getattr__(self, name): return self[name]
    def __setattr__(self, name, value): self[name] = value
    def __delattr__(self, name): del self[name]

def Dict_to_dict(cfg):
    """
    transform a Dict into a dict
    """
    ty = type(cfg)
    cfg = dict(cfg)
    for k,i in cfg.items():
        if type(i)==ty:
            cfg[k] = Dict_to_dict(cfg[k])
    return cfg

def save_config(path, cfg):
    """
    save a configuration in a yaml file.
    path must thus contains a yaml extension.
    example: path='logs/test.yaml'
    """
    cfg = Dict_to_dict(cfg)
    with open(path, "w") as f:
        yaml.dump(cfg, f, sort_keys=False)

def nested_dict_pairs_iterator(dic):
    ''' This function accepts a nested dictionary as argument
        and iterate over all values of nested dictionaries
        stolen from: https://thispointer.com/python-how-to-iterate-over-nested-dictionary-dict-of-dicts/ 
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

def popupmsg(msg):
    popup = tk.Tk()
    popup.wm_title("!")
    label = ttk.Label(popup, text=msg)
    #popup.geometry('300x100')
    popup.minsize(300,100)
    label.pack(pady=10, padx=10)
    B1 = ttk.Button(popup, text="Okay", command = popup.destroy)
    B1.pack(side="bottom",pady=10)
    popup.mainloop()
    
 
def replace_line_single(line, key, value):
 
    if key==line[:len(key)]:
        assert line[len(key):len(key)+3]==" = ", "[Error] Invalid line. A valid line must contains \' = \'. Line:"+line
        line = line[:len(key)]
        
        # if value is string then we add brackets
        line += " = "
        if type(value)==str: 
            line += "\'" + value + "\'"
        elif type(value)==np.ndarray:
            line += str(value.tolist())
        else:
            line += str(value)
    return line

def replace_line_multiple(line, dic):

    for key, value in dic.items():
        line = replace_line_single(line, key, value)
    return line

    
def save_python_config(
    config_dir,
    base_config = None,
    **kwargs,
    ):
    import shutil
    import fileinput
    from datetime import datetime

    # create the config dir if needed
    if not os.path.exists(config_dir):
        os.makedirs(config_dir, exist_ok=True)

    # copy default config file or use the one given by the user
    if base_config == None:
        try:
            from biom3d import config_default
            config_path = shutil.copy(config_default.__file__, config_dir) 
        except:
            print("[Error] Please provide a base config file or install biom3d.")
            raise RuntimeError
    else: 
        config_path = base_config

    # rename it with date included
    current_time = datetime.now().strftime("%Y%m%d-%H%M%S")

    # if DESC is in kwargs, then it will be used to rename the config file
    basename = os.path.basename(config_path) if "DESC" not in kwargs.keys() else kwargs['DESC']+'.py'
    new_config_name = os.path.join(config_dir, current_time+"-"+basename)
    os.rename(config_path, new_config_name)

    # edit the new config file with the auto-config values
    with fileinput.input(files=(new_config_name), inplace=True) as f:
        for line in f:
            # edit the line
            line = replace_line_multiple(line, kwargs)
            # write back in the input file
            print(line, end='') 
    return new_config_name   

def config_to_type(cfg, new_type):
    """Change config type to a new type. This function is recursive and can be use to change the type of nested dictionaries. 
    """
    old_type = type(cfg)
    cfg = new_type(cfg)
    for k,i in cfg.items():
        if type(i)==old_type:
            cfg[k] = config_to_type(cfg[k], new_type)
    return cfg
def load_python_config(config_path):
    import importlib.util
    import sys
    spec = importlib.util.spec_from_file_location("config", config_path)
    config = importlib.util.module_from_spec(spec)
    sys.modules["config"] = config
    spec.loader.exec_module(config)
    return config_to_type(config.CONFIG, Dict) # change type from config.Dict to Dict
   
#----------------------------------------------------------------------------
# File dialog

class FileDialog(ttk.Frame):
    def __init__(self, *arg, mode='folder', textEntry="", **kw):
        super(FileDialog, self).__init__(*arg, **kw)
        assert mode in ['folder','file']

        self.filenameText = StringVar()
        self.set(textEntry)
        self.filename = ttk.Entry(self, textvariable=self.filenameText)
        self.filename.grid(column=1,row=1, sticky=(W,E))
        self.grid_columnconfigure(1, weight=1)

        self.command=self.openfolder if mode=='folder' else self.openfile
        self.button = ttk.Button(self,text="Browse", style="train_button.TLabel", width=8, command=self.command)
        self.button.grid(column=2,row=1, sticky=(W), ipady=4, padx=5)

    def set(self, text):
        # Set text entry
        self.filenameText.set(text)

    def openfolder(self):
        text = filedialog.askdirectory(initialdir =  ".", title = "Select A Folder")
        self.filenameText.set(text)
    
    def openfile(self):
        text = filedialog.askopenfilename(initialdir =  ".", title = "Select A File")
        self.filenameText.set(text)
    
    def get(self):
        return self.filenameText.get()


#----------------------------------------------------------------------------
# Preprocess & Train tab

class TrainFolderSelection(ttk.LabelFrame):
    def __init__(self, preprocess_tab=None, *arg, **kw):
        super(TrainFolderSelection, self).__init__(*arg, **kw)

        # Define elements
        # use preprocessing values
        train_button_css = ttk.Style()
        train_button_css.configure("train_button.TLabel", background = '#CD5C5C', foreground = 'white', font=('Helvetica', 9), width = 30, borderwidth=3, focusthickness=7, relief='raised', focuscolor='none', anchor='c', height= 15)
        self.send_data_button = ttk.Button(self, text="Send Dataset",width=29,style="train_button.TLabel",  command=self.send_data)
        self.preprocess_tab = preprocess_tab

        ## image folder
        self.label1 = ttk.Label(self, text="Select the folder containing the images:", anchor="sw", background='white')
        ## mask folder
        self.label2 = ttk.Label(self, text="Select the folder containing the masks:", anchor="sw", background='white')

        if REMOTE:

            self.label0 = ttk.Label(self, text="Send a new Dataset :", anchor="sw", background='white')
            self.img_outdir = FileDialog(self, mode='folder', textEntry='')        
            self.msk_outdir = FileDialog(self, mode='folder', textEntry="")
            self.send_data_label = ttk.Label(self, text="Define a unique name below to your Dataset :")
            self.send_data_name = StringVar(value="nucleus_0001")
            self.send_data_entry = ttk.Entry(self, textvariable=self.send_data_name)
            

    
          
        else:
            self.img_outdir = FileDialog(self, mode='folder', textEntry="/home/safarbatis/chocolate-factory/data/raw")
            self.msk_outdir = FileDialog(self, mode='folder', textEntry="/home/safarbatis/chocolate-factory/data/seg")            
       

        # Position elements

        self.label1.grid(column=0, row=1, sticky=W, pady=5)
        
        if REMOTE:
            self.label0.grid(column=0,row=0, sticky=W, pady=2)
            self.img_outdir.grid(column=0, row=2, sticky=(W,E))
            self.label2.grid(column=0,row=3, sticky=W, pady=2)
            self.msk_outdir.grid(column=0,row=4, sticky=(W,E))
            self.send_data_label.grid(column=0, row=7, sticky=(W,E), pady=2)
            self.send_data_entry.grid(column=0,row=8, sticky=(W,E))
            self.send_data_button.grid(column=0, row=9, pady=5,ipady=4,)
            
        else:
            self.img_outdir.grid(column=0, row=2, sticky=(W,E))
            self.label2.grid(column=0,row=3, sticky=W, pady=7)
            self.msk_outdir.grid(column=0,row=4, sticky=(W,E))

        
        
        # Configure columns
        self.columnconfigure(0, weight=1)

        for i in range(6):
            self.rowconfigure(i, weight=1)

    def use_preprocessing(self):
        if REMOTE:
            # update the list of data directories
            _,stdout,_ = REMOTE.exec_command('ls {}/data'.format(MAIN_DIR))
            self.data_list = [e.replace('\n','') for e in stdout.readlines()]

            # update option menu list 
            self.data_dir_option_menu.set_menu(self.data_list[0], *self.data_list)
            
            self.data_dir.set(self.preprocess_tab.send_data_name.get())
            self._update_data_dir()
        else:
            self.img_outdir.set(self.img_outdir.get())
            self.msk_outdir.set(self.msk_outdir.get())

        self.num_classes.set(self.num_classes.get())
    
    def send_data(self):
        ftp = REMOTE.open_sftp()

        # copy folders 
        remote_dir_img = "{}/data/{}/img".format(MAIN_DIR,self.send_data_name.get())
        remote_dir_msk = "{}/data/{}/msk".format(MAIN_DIR,self.send_data_name.get())
        ftp_put_folder(ftp, localpath=self.img_outdir.get(), remotepath=remote_dir_img)
        ftp_put_folder(ftp, localpath=self.msk_outdir.get(), remotepath=remote_dir_msk)
        popupmsg("Data sent!")
     
   
class ConfigFrame(ttk.LabelFrame):
    def __init__(self, train_folder_selection=None, *arg, **kw):
        super(ConfigFrame, self).__init__(*arg, **kw)

        # widgets definitions
        willy1 = ttk.Style()
        willy1.configure("auto_confing_button.TLabel", background = '#76D7C4', foreground = 'black', width = 25, borderwidth=3, focusthickness=7, focuscolor='red', relief="raised" , anchor='c')
        if REMOTE:
            # get dataset list
            _,stdout,_ = REMOTE.exec_command('ls {}/data'.format(MAIN_DIR))
            self.data_list = [e.replace('\n','') for e in stdout.readlines()]
            # define the dropdown menu
            if(len(self.data_list) == 0):
                self.data_dir = StringVar(value="Empty")
            else:   
                #self.data_dir = StringVar(value=self.data_list[0])
                self.data_dir = StringVar(value="Choose a dataset")

            self.data_dir_option_menu = ttk.OptionMenu(self, self.data_dir, self.data_dir.get(), *self.data_list, command= self.option_selected)


            self.label4 = ttk.Label(self, text="Select the Dataset to preprocess :", anchor="sw", background='white')
            self.data_dir.trace("w", self.option_selected)
            self.dataset_update()        
            self.button_dataset_list = ttk.Button(self, text="Update", command=self.dataset_update)
        
        self.builder_name_label = ttk.Label(self, text="Set a name for the builder folder (folder containing your future model):")
        self.builder_name = StringVar(value="unet_example")
        self.builder_name_entry = ttk.Entry(self, textvariable=self.builder_name)

        # number of classes
        self.num_classes_label = ttk.Label(self, text="Enter the number of classes: "+"."*screen_width, anchor="sw", background='white')
        self.num_classes = IntVar(value=1)
        self.classes = ttk.Entry(self, width=5,textvariable=self.num_classes)
        
        
        self.auto_config_button = ttk.Button(self, text="Auto-configure", style='train_button.TLabel',width =29,command=self.auto_config)
        self.img_outdir = train_folder_selection.img_outdir
        self.msk_outdir = train_folder_selection.msk_outdir
        

        self.auto_config_finished = ttk.Label(self, text="")

        self.num_epochs_label = ttk.Label(self, text='Number of epochs: '+"."*screen_width)
        self.num_epochs = IntVar(value=100)
        self.num_epochs_entry = ttk.Entry(self, width=4, textvariable=self.num_epochs)
        

        self.batch_size_label = ttk.Label(self, text='Batch size (int): '+"."*screen_width)
        self.batch_size = IntVar(value=2)
        self.batch_size_entry = ttk.Entry(self, width=4, textvariable=self.batch_size)

        self.patch_size_label = ttk.Label(self, text='Patch size ([int int int]): '+"."*screen_width)
        self.patch_size1 = StringVar(value="128")
        self.patch_size_entry1 = ttk.Entry(self, width=4, textvariable=self.patch_size1)
        self.patch_size2 = StringVar(value="128")
        self.patch_size_entry2 = ttk.Entry(self, width=4, textvariable=self.patch_size2)
        self.patch_size3 = StringVar(value="128")
        self.patch_size_entry3 = ttk.Entry(self, width=4, textvariable=self.patch_size3)
        self.patch_size = [int(self.patch_size1.get()), int(self.patch_size2.get()), int(self.patch_size3.get())]
       

        self.aug_patch_size_label = ttk.Label(self, text='Augmentation patch size ([int int int]): '+"."*screen_width)
        self.aug_patch_size1 = StringVar(value="160")
        self.aug_patch_size_entry1 = ttk.Entry(self, width=4, textvariable=self.aug_patch_size1)
        self.aug_patch_size2 = StringVar(value="120")
        self.aug_patch_size_entry2 = ttk.Entry(self, width=4, textvariable=self.aug_patch_size2)
        self.aug_patch_size3 = StringVar(value="160")
        self.aug_patch_size_entry3 = ttk.Entry(self, width=4, textvariable=self.aug_patch_size3)
        self.aug_patch_size = [int(self.aug_patch_size1.get()), int(self.aug_patch_size2.get()), int(self.aug_patch_size3.get())]
        
        self.num_pools_label = ttk.Label(self, text='Number of pool in the U-Net model ([int int int]): '+"."*screen_width)
        self.num_pools1 = StringVar(value="5")
        self.num_pools_entry1 = ttk.Entry(self, width=4, textvariable=self.num_pools1)
        self.num_pools2 = StringVar(value="5")
        self.num_pools_entry2 = ttk.Entry(self, width=4, textvariable=self.num_pools2)
        self.num_pools3 = StringVar(value="5")
        self.num_pools_entry3 = ttk.Entry(self, width=4, textvariable=self.num_pools3)
        self.num_pools = [int(self.num_pools1.get()), int(self.num_pools2.get()), int(self.num_pools3.get())]

        # place widgets
        if REMOTE:
            self.data_dir_option_menu.grid(column=0,sticky=E,padx=75 ,row=0, pady=2)
            self.label4.grid(column=0,row=0, sticky=W, pady=2)
            self.button_dataset_list.grid(column=0,sticky=E, row=0, pady=2)
        self.builder_name_label.grid(column=0, row=1, sticky=(W,E), ipady=4,pady=3)
        self.builder_name_entry.grid(column=0, row=2,ipadx=213,ipady=4,pady=3,sticky=(W,E))
        self.num_classes_label.grid(column=0,row=3, sticky=W, pady=2)
        self.classes.grid(column=0,row=3,pady=5,sticky=E)
        self.auto_config_button.grid(column=0, columnspan=4,row=4 ,ipady=4, pady=2,)
        self.auto_config_finished.grid(column=0, row=5, columnspan=2, sticky=(W,E))

        self.num_epochs_label.grid(column=0, row=6, sticky=(W))
        self.num_epochs_entry.grid(column=0, row=6, padx= 0, sticky=E)

        self.batch_size_label.grid(column=0, row=7, sticky=(W,E))
        self.batch_size_entry.grid(column=0, row=7, padx= 0, sticky=E)
        
        self.patch_size_label.grid(column=0, row=8, sticky=(W,E))
        self.patch_size_entry1.grid(column=0, row=8,padx= 60, sticky=E)
        self.patch_size_entry2.grid(column=0, row=8,padx= 30, sticky=E)
        self.patch_size_entry3.grid(column=0, row=8, padx= 0, sticky=E)

        self.aug_patch_size_label.grid(column=0, row=9, sticky=(W,E))
        self.aug_patch_size_entry1.grid(column=0, row=9,padx= 60, sticky=E)
        self.aug_patch_size_entry2.grid(column=0, row=9, padx= 30,sticky=E)
        self.aug_patch_size_entry3.grid(column=0, row=9,padx= 0, sticky=E)

        self.num_pools_label.grid(column=0, row=10, sticky=(W,E))
        self.num_pools_entry1.grid(column=0, row=10,padx= 60, sticky=E)
        self.num_pools_entry2.grid(column=0, row=10, padx= 30,sticky=E)
        self.num_pools_entry3.grid(column=0, row=10,padx= 0, sticky=E)

        # grid config
        self.columnconfigure(0, weight=1)
        for i in range(10):
            self.rowconfigure(i, weight=1)
    def option_selected(self, *args):
        global selected_dataset
        selected_dataset = self.data_dir.get()
        print("Selected option:", selected_dataset)
        return selected_dataset

    def dataset_update(self):
        # get updated dataset list from remote
        _, stdout, _ = REMOTE.exec_command('ls {}/data'.format(MAIN_DIR))
        self.data_list = [e.replace('\n','') for e in stdout.readlines()]

        # clear existing menu
        self.data_dir_option_menu['menu'].delete(0, 'end')

        # add new options to menu
        for option in self.data_list:
            self.data_dir_option_menu['menu'].add_command(
                label=option,
                command=lambda opt=option: self.data_dir.set(opt))
        
        
    def str2list(self, string):
        """
        convert a string like '[5 5 5]' into list of integers
        we remove first and last element, as they are supposed to be '[' and ']' symbols.
        """
        # remove first and last element
        return [int(e) for e in string[1:-1].split(' ') if e!='']
    def str2list2(self, string):
        """
        convert a string like '[5, 5, 5]' into list of integers
        we remove first and last element, as they are supposed to be '[' and ']' symbols.
        """
        # remove first and last element
        return [int(e) for e in string[1:-1].split(', ') if e!='']
    def auto_config(self):
        self.auto_config_finished.config(text="Auto-configuration, please wait...")
        global config_path 
        global img_dir_train
        global msk_dir_train
        global fg_dir_train
        if REMOTE:
            # preprocessing
            _,stdout,stderr=REMOTE.exec_command("source {}/bin/activate; cd {};  python -m biom3d.preprocess --img_dir data/{}/img --msk_dir data/{}/msk --num_classes {} --desc {} --remote".format(VENV,MAIN_DIR, selected_dataset, selected_dataset,self.classes.get(),self.builder_name_entry.get()))  
            auto_config_results = stdout.readlines()
            auto_config_results = [e.replace('\n','') for e in auto_config_results]
         
            img_dir_train = "data/{}/img_out".format(selected_dataset)
            msk_dir_train = "data/{}/msk_out".format(selected_dataset)
            fg_dir_train = "data/{}/fg_out".format(selected_dataset)
            # error management
            if len(auto_config_results)!=10:
               print("[Error] Auto-config error:", auto_config_results)
               popupmsg("[Error] Auto-config error: "+ str(auto_config_results))
            while True:
                line = stderr.readline()
                if not line:
                    break
                print(line, end="")
           
           # get auto config results
            reversed_auto_config_results = auto_config_results[::-1]
        
            batch = reversed_auto_config_results[4]
            patch = self.str2list(reversed_auto_config_results[3])
            aug_patch = self.str2list2(reversed_auto_config_results[2])
            pool = self.str2list(reversed_auto_config_results[1])
            config_path = reversed_auto_config_results[0]
    
            print(batch,patch,pool,aug_patch,config_path)
          
            
        else: 
            # Preprocessing & autoconfiguration    
            config_path=auto_config_preprocess(img_dir=self.img_outdir.get(),
            msk_dir=self.msk_outdir.get(),
            desc=self.builder_name_entry.get(),
            num_classes=self.num_classes.get()+1,
            remove_bg=False, use_tif=False,
            config_dir="config/",
            base_config=None,
                
            )
            # Read the config file
            self.train_parameters = adaptive_load_config(config_path)
            
            batch= self.train_parameters.BATCH_SIZE
            aug_patch= self.train_parameters.AUG_PATCH_SIZE
            patch= self.train_parameters.PATCH_SIZE
            pool = self.train_parameters.NUM_POOLS
            
            img_dir_train = self.train_parameters.IMG_DIR
            msk_dir_train = self.train_parameters.MSK_DIR
            fg_dir_train = self.train_parameters.FG_DIR
         
            print("Train Parameters :  ",batch,aug_patch,patch,pool,config_path)
           
        
        # Update Config Cells in Gui
        self.batch_size.set(batch)
        
        self.aug_patch_size= aug_patch
        self.aug_patch_size1.set(aug_patch[0])
        self.aug_patch_size2.set(aug_patch[1])
        self.aug_patch_size3.set(aug_patch[2])
        
        self.patch_size = patch
        self.patch_size1.set(patch[0])
        self.patch_size2.set(patch[1])
        self.patch_size3.set(patch[2])
        
        self.num_pools = pool
        self.num_pools1.set(pool[0])
        self.num_pools2.set(pool[1])
        self.num_pools3.set(pool[2])
        
        if REMOTE:
            self.auto_config_finished.config(text="Auto-configuration done! \n" )
            popupmsg("Auto-configuration done!")
        else :
            self.auto_config_finished.config(text="Auto-configuration done! and saved in config folder : \n" +config_path)
            popupmsg("Auto-configuration done! and saved in config folder : \n" +config_path)

class TrainTab(ttk.Frame):
    def __init__(self, preprocess_tab=None, *arg, **kw):
        super(TrainTab, self).__init__(*arg, **kw)
        global new_config_path
        self.folder_selection = TrainFolderSelection(preprocess_tab=preprocess_tab, master=self, text="Preprocess", padding=[10,10,10,10])
        self.config_selection = ConfigFrame(train_folder_selection=self.folder_selection, master=self, text="Training configuration", padding=[10,10,10,10])
        self.train_button = ttk.Button(self, text="Start", style="train_button.TLabel", width =29, command=self.train)
        self.train_done = ttk.Label(self, text="")

        # set default values of train folders with the ones used for preprocess tab

        self.folder_selection.grid(column=0,row=0,sticky=(N,W,E), pady=3)
        self.config_selection.grid(column=0,row=1,sticky=(N,W,E), pady=20)
        self.train_button.grid(column=0, row=2, padx=15, ipady=4, pady= 2, sticky=(N))
        self.train_done.grid(column=0, row=3, sticky=W)

    
        self.columnconfigure(0, weight=1)
        for i in range(4):
            self.rowconfigure(i, weight=1)
    
    def str2list(self, string):
        """
        convert a string like '[5 5 5]' into list of integers
        we remove first and last element, as they are supposed to be '[' and ']' symbols.
        """
        # remove first and last element
        return [int(e) for e in string[1:-1].split(' ') if e!='']

    def line_buffered(self, f):
        line_buf = ""
        while not f.channel.exit_status_ready():
            line_buf += str(f.read(1))
            if line_buf.endswith('\n'):
                yield line_buf
                line_buf = ''

    def train(self):
        if REMOTE:
            import tempfile
            tempdir = tempfile.mkdtemp(prefix="tmp-config-")
    
            saved_config = tempdir+"/tmp-config.py"
            ftp = REMOTE.open_sftp()
          
            ftp.get(MAIN_DIR+"/"+config_path, saved_config)
            
            cfg = load_python_config(saved_config)
            ftp.close()
        else :   
            cfg = load_python_config(config_path)
       
       
       # test if aug_patch_size is greater than patch_size 
        self.config_selection.patch_size = [int(self.config_selection.patch_size1.get()), int(self.config_selection.patch_size2.get()), int(self.config_selection.patch_size3.get())]
        self.config_selection.aug_patch_size = [int(self.config_selection.aug_patch_size1.get()), int(self.config_selection.aug_patch_size2.get()), int(self.config_selection.aug_patch_size3.get())]
       
        for i in range(len(self.config_selection.patch_size)):
         if self.config_selection.patch_size[i] > self.config_selection.aug_patch_size[i]:
             popupmsg(f" ERREUR ! Le patch size N°{i} est inferieur a l'élément {i} de l'augmentation patch size !")
       
        # set the configuration
        
        cfg.IMG_DIR = img_dir_train

        cfg = nested_dict_change_value(cfg, 'img_dir', cfg.IMG_DIR)

        cfg.MSK_DIR = msk_dir_train
        cfg = nested_dict_change_value(cfg, 'msk_dir', cfg.MSK_DIR)
        
        cfg.FG_DIR = fg_dir_train
        cfg = nested_dict_change_value(cfg, 'fg_dir', cfg.FG_DIR)

        cfg.DESC = self.config_selection.builder_name.get()
        
        cfg.NUM_CLASSES = self.config_selection.num_classes.get()
        cfg = nested_dict_change_value(cfg, 'num_classes', cfg.NUM_CLASSES)
       
        cfg.NB_EPOCHS = self.config_selection.num_epochs.get()
        cfg = nested_dict_change_value(cfg, 'nb_epochs', cfg.NB_EPOCHS)
        
        cfg.BATCH_SIZE = self.config_selection.batch_size.get()
        cfg = nested_dict_change_value(cfg, 'batch_size', cfg.BATCH_SIZE)
       
        
        cfg.PATCH_SIZE = self.config_selection.patch_size
        cfg = nested_dict_change_value(cfg, 'patch_size', cfg.PATCH_SIZE)
        
        
        cfg.AUG_PATCH_SIZE = self.config_selection.aug_patch_size
        cfg = nested_dict_change_value(cfg, 'aug_patch_size', cfg.AUG_PATCH_SIZE)

        self.config_selection.num_pools = [int(self.config_selection.num_pools1.get()), int(self.config_selection.num_pools2.get()), int(self.config_selection.num_pools3.get())]
        cfg.NUM_POOLS = self.config_selection.num_pools
        cfg = nested_dict_change_value(cfg, 'num_pools', cfg.NUM_POOLS)
        
        if REMOTE:
            # if remote store the config file in a temp file
    
            new_config_path = save_python_config(config_dir=tempdir,
                                                 base_config=saved_config,
            IMG_DIR=cfg.IMG_DIR,
            MSK_DIR=cfg.MSK_DIR,
            FG_DIR=cfg.FG_DIR,
            NUM_CLASSES=cfg.NUM_CLASSES,
            BATCH_SIZE=cfg.BATCH_SIZE,
            AUG_PATCH_SIZE=cfg.AUG_PATCH_SIZE,
            PATCH_SIZE=cfg.PATCH_SIZE,
            NUM_POOLS=cfg.NUM_POOLS,
            NB_EPOCHS=cfg.NB_EPOCHS)
            # copy it
            ftp = REMOTE.open_sftp()
            ftp.put(new_config_path, MAIN_DIR+"/config1.py")
            ftp.close()
            # delete the temp file
            os.remove(new_config_path)


            import threading
            def train_nohup():
                
                # run the training and store the output in an output file 
                _,stdout,stderr=REMOTE.exec_command("source {}/bin/activate; cd {}; nohup  python -m biom3d.train --config config1.py &".format(VENV,MAIN_DIR))
                
                # print the stdout continuously
                while True:
                    line = stdout.readline()
                    if not line:
                        break
                    print(line, end="")
                while True:
                    line = stderr.readline()
                    if not line:
                        break
                    print(line, end="")
                
        
                
            worker = threading.Thread(target=train_nohup)
            worker.start()
            time.sleep(3)
            
            
            def plot():
                import matplotlib.pyplot as plt
                import pandas as pd
                import os
                import time  
                plt.switch_backend('agg')            
                # Function to update the plot
                def update_plot(csv_file):                   
                    data = pd.read_csv(csv_file)
                    plt.clf()  # Clear the current plot
                    plt.plot(data['epoch'], data['train_loss'], label='Train loss')
                    plt.plot(data['epoch'], data['val_loss'], label ='Validation loss')
                    plt.xlabel('Epoch')
                    plt.ylabel('Loss')
                    plt.title('Learning Curves')
                    plt.grid(True)
                    plt.legend()
                   #plt.pause(0.1)  # Pause for a short duration to allow for updating               
                    plt.savefig('Learning_curves_plot.png')
                    
                # CSV file path
                csv_file = os.path.join("plots/log.csv")

                # Get initial modification timestamp
                prev_mod_time = os.path.getmtime(csv_file)

                # Continuously update the plot
                while True:
                    curr_mod_time = os.path.getmtime(csv_file)
                    if curr_mod_time > prev_mod_time:
                        prev_mod_time = curr_mod_time
                        update_plot(csv_file)
                    time.sleep(1)  # Wait for 1 second before checking again
                    if not worker.is_alive():
                        break

        
            def get_logs():
                while True:
                    time.sleep(2)
                    # get the last folder modified/created
                    _,stdout,stderr=REMOTE.exec_command("ls -td {}/logs/*/ | head -1".format(MAIN_DIR))
                    line= stdout.readline()
                
                    # connect to remote
                    ftp = REMOTE.open_sftp()

                    # remote directory
                    remotedir =line.rstrip()+"/log"
                    
                    # create local dir if it does not exist already
                    localdir = os.path.join("plots/")
                    if not os.path.exists(localdir):
                        os.makedirs(localdir, exist_ok=True)
                    
                    # copy files from remote to local
                    ftp_get_folder(ftp, remotedir, localdir)
                    if not worker.is_alive():
                        break
            
            
            
            
            
            worker2 = threading.Thread(target=get_logs)
            worker2.start()
            time.sleep(3)
            worker3 = threading.Thread(target=plot)
            worker3.start()
     
                
             
            worker.join()
            if not worker.is_alive() : popupmsg(" Training Done ! ")
            
            #worker2.join()
            #worker3.join()
            #get_logs()
            #plot()

            
            
        else:  
            # save the new config file
            
            new_config_path = save_python_config(
            config_dir="config/",
            base_config=config_path,
            IMG_DIR=cfg.IMG_DIR,
            MSK_DIR=cfg.MSK_DIR,
            NUM_CLASSES=self.config_selection.num_classes.get(),
            BATCH_SIZE=cfg.BATCH_SIZE,
            AUG_PATCH_SIZE=cfg.AUG_PATCH_SIZE,
            PATCH_SIZE=cfg.PATCH_SIZE,
            NUM_POOLS=cfg.NUM_POOLS,
            NB_EPOCHS=cfg.NB_EPOCHS
            
            )
        
            if not torch.cuda.is_available():
               popupmsg("  No GPU detected, the training might take a longer time ")
        
            # run the training
            train(config=new_config_path)
            popupmsg(" Training Done ! ")
            self.train_done.config(text="Training done!")
              

        

#----------------------------------------------------------------------------
# Precition tab

class InputDirectory(ttk.LabelFrame):
    def __init__(self, *arg, **kw):
        super(InputDirectory, self).__init__(*arg, **kw)

        self.input_folder_label = ttk.Label(self, text="Select a folder containing images for prediction:")
        self.input_folder_label.grid(column=0, row=0, sticky=(W,E))

        if REMOTE: 
            # define the dropdown menu
            _,stdout,_ = REMOTE.exec_command('ls {}/data/to_pred'.format(MAIN_DIR))     # Where should i search ??
            self.data_list = [e.replace('\n','') for e in stdout.readlines()]
            if(len(self.data_list) == 0):
                self.data_dir = StringVar(value="Empty")
            else:   
                self.data_dir = StringVar(value=self.data_list[0])
            self.data_dir_option_menu = ttk.OptionMenu(self, self.data_dir, self.data_dir.get(), *self.data_list)

            # or send the dataset to server
            self.send_data_label = ttk.Label(self, text="Or send a new dataset of raw images to the server:")
            self.send_data_folder = FileDialog(self, mode='folder', textEntry="data/to_pred")
            self.send_data_button = ttk.Button(self, width=29, style="train_button.TLabel", text="Send data", command=self.send_data)


            self.data_dir_option_menu.grid(column=0, row=0, padx=12,sticky=(E))
            self.send_data_label.grid(column=0, row=2, sticky=(W,E))
            self.send_data_folder.grid(column=0, row=3, sticky=(W,E))
            self.send_data_button.grid(column=0, row=4, ipady=4,pady=5,)

            self.columnconfigure(0, weight=1)
            for i in range(5):
                self.rowconfigure(i, weight=1)
        else:
            
            self.data_dir = FileDialog(self, mode='folder', textEntry=os.path.join('data', 'to_pred'))
            self.data_dir.grid(column=0, row=1, sticky=(W,E))

            self.columnconfigure(0, weight=1)
            for i in range(2):
                self.rowconfigure(i, weight=1)

    def send_data(self):
        # send data to server
        ftp = REMOTE.open_sftp()
        remotepath="{}/data/to_pred/{}".format(MAIN_DIR, os.path.basename(self.send_data_folder.get()))
        ftp_put_folder(ftp, localpath=self.send_data_folder.get(), remotepath=remotepath)

        # update the dropdown menu (and select the new dataset automatically)
        _,stdout,_ = REMOTE.exec_command('ls {}/data/to_pred'.format(MAIN_DIR))
        self.data_list = [e.replace('\n','') for e in stdout.readlines()]
        self.data_dir_option_menu.set_menu(self.data_list[0], *self.data_list)
        self.data_dir.set(os.path.basename(self.send_data_folder.get()))
        popupmsg("Data sent !")

class Connect2Omero(ttk.LabelFrame):
    def __init__(self, *arg, **kw):
        super(Connect2Omero, self).__init__(*arg, **kw)

        # widgets definitions
        self.hostname_label = ttk.Label(self, text='Omero server address:')
        self.hostname = StringVar(value="omero.igred.fr")
        self.hostname_entry = ttk.Entry(self, textvariable=self.hostname)

        self.username_label = ttk.Label(self, text='User name:')
        self.username = StringVar(value="")
        self.username_entry = ttk.Entry(self, textvariable=self.username)

        self.password_label = ttk.Label(self, text='Password:')
        self.password = StringVar(value="")
        self.password_entry = ttk.Entry(self, textvariable=self.password, show='*')

        # place widgets
        self.hostname_label.grid(column=0, row=0, sticky=(W,E))
        self.hostname_entry.grid(column=1, row=0, sticky=(W,E))

        self.username_label.grid(column=0, row=1, sticky=(W,E))
        self.username_entry.grid(column=1, row=1, sticky=(W,E))

        self.password_label.grid(column=0, row=2, sticky=(W,E))
        self.password_entry.grid(column=1, row=2, sticky=(W,E))

        # grid config
        self.columnconfigure(0, weight=1)
        self.columnconfigure(1, weight=5)
        for i in range(3):
            self.rowconfigure(i, weight=1)

class OmeroDataset(ttk.LabelFrame):
    def __init__(self, *arg, **kw):
        super(OmeroDataset, self).__init__(*arg, **kw)

    
        self.label_id = ttk.Label(self, text="Input Dataset ID:")
        self.id = StringVar(value="19699")
        self.id_entry = ttk.Entry(self, textvariable=self.id)

   
        
        self.label_id.grid(column=0, row=0, sticky=(W,E))
        self.id_entry.grid(column=1,row=0,sticky=(W,E))

        
        self.columnconfigure(0, weight=1)
        self.columnconfigure(1, weight=5)
        self.rowconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)
class ModelSelection(ttk.LabelFrame):
    def __init__(self, *arg, **kw):
        super(ModelSelection, self).__init__(*arg, **kw)

        # Define elements
        # get model list
        if REMOTE:
            _,stdout,_ = REMOTE.exec_command('ls {}/logs'.format(MAIN_DIR))
            self.logs_list = [e.replace('\n','') for e in stdout.readlines()]
            
            # define the dropdown menu    
            if(len(self.logs_list) == 0):
                self.logs_dir = StringVar(value="Empty")
            else:   
                self.logs_dir = StringVar(value=self.logs_list[0])
            self.logs_dir_option_menu = ttk.OptionMenu(self, self.logs_dir, self.logs_dir.get(), *self.logs_list)
            self.button_update_list = ttk.Button(self, text="Update", command=self._update_logs_list)
            
            self.logs_dir_option_menu.grid(column=0, row=0, sticky=(W,E))
            self.button_update_list.grid(column=1, row=0, padx=5,sticky=(W,E))
            
            self.columnconfigure(0, weight=10)
            self.columnconfigure(1, weight=1)
            self.rowconfigure(0, weight=1)

        else: 
            ## build folder
            self.label1 = ttk.Label(self, text="Select the folder containing the build:", anchor="sw", background='white')
            self.logs_dir = FileDialog(self, mode='folder', textEntry='logs/')

            self.logs_dir.grid(column=0, row=0, sticky=(W,E))
            self.columnconfigure(0, weight=1)
            self.rowconfigure(0, weight=1)
        
    def _update_logs_list(self):
        _,stdout,_ = REMOTE.exec_command('ls {}/logs'.format(MAIN_DIR))
        self.logs_list = [e.replace('\n','') for e in stdout.readlines()]
        self.logs_dir_option_menu.set_menu(self.logs_list[0], *self.logs_list)
        popupmsg("Models list updated !")

class OutputDirectory(ttk.LabelFrame):
    def __init__(self, *arg, **kw):
        super(OutputDirectory, self).__init__(*arg, **kw)

        if REMOTE: 
            # if remote, only print an information message indicating where to find the prediction folder on the server.
            self.default_label = ttk.Label(self, text="(Optional) Select a local folder to download the predictions:")
            self.data_dir = FileDialog(self, mode="folder", textEntry="")
            self.default_label.grid(column=0, row=0, sticky=(W,E))
            self.data_dir.grid(column=0, row=1, sticky=(W,E))

            self.columnconfigure(0, weight=1)
            for i in range(2):
                self.rowconfigure(i, weight=1)
        else:
            self.data_dir_label = ttk.Label(self, text="Select a folder for the upcoming predictions:")
            self.data_dir = FileDialog(self, mode='folder', textEntry=os.path.join('data', 'pred'))
            
            self.data_dir_label.grid(column=0, row=0, sticky=(W,E))
            self.data_dir.grid(column=0, row=1, sticky=(W,E))

            self.columnconfigure(0, weight=1)
            for i in range(2):
                self.rowconfigure(i, weight=1)
class OmeroUpload(ttk.LabelFrame):
    def __init__(self,*arg, **kw):
        super(OmeroUpload, self).__init__(*arg, **kw)
        self.upload_label= ttk.Label(self, text="Send predictions to OMERO : ")
       
        # widgets definitions
        self.hostname_label = ttk.Label(self, text='Omero server address:')
        self.hostname = StringVar(value="omero.igred.fr")
        self.hostname_entry = ttk.Entry(self, textvariable=self.hostname)

        self.username_label = ttk.Label(self, text='User name:')
        self.username = StringVar(value="")
        self.username_entry = ttk.Entry(self, textvariable=self.username)

        self.password_label = ttk.Label(self, text='Password:')
        self.password = StringVar(value="")
        self.password_entry = ttk.Entry(self, textvariable=self.password, show='*')

        self.dataset_label = ttk.Label(self, text='Select a folder to save the prediction  :')
        
        self.prediction_folder_label = ttk.Label(self, text='Prediction Folder:')
        self.prediction_folder= FileDialog(self, mode='folder', textEntry="data/pred")
        
        self.upload_project_label= ttk.Label(self, text="Project ID : ") #change to project id and send dataset name
        self.project_id= StringVar(value="12906")
        self.upload_project_entry= ttk.Entry(self,textvariable=self.project_id)
        
        self.upload_dataset_label= ttk.Label(self, text="Select a name for your dataset : ") #change to project id and send dataset name
        self.dataset_name= StringVar(value="Biom3d_pred")
        self.dataset_name_entry= ttk.Entry(self,textvariable=self.dataset_name)
        if REMOTE :
            # define the dropdown menu
            _,stdout,_ = REMOTE.exec_command('ls {}/data/pred'.format(MAIN_DIR))
            self.data_list = [e.replace('\n','') for e in stdout.readlines()]
            if(len(self.data_list) == 0):
                    self.data_dir = StringVar(value="Empty")
            else:   
                self.data_dir = StringVar(value=self.data_list[0])
            self.data_dir_option_menu = ttk.OptionMenu(self, self.data_dir, self.data_dir.get(), *self.data_list)
            self.button_update_list = ttk.Button(self, text="Update", command=self._update_pred_list)
            self.dataset_label = ttk.Label(self, text='Select the dataset to send :')
        self.send_data_omero = ttk.Button(self, width=29,text="Send to Omero", style="train_button.TLabel", command=self.send_to_omero)
        # place widgets
        self.hostname_label.grid(column=0, row=0, sticky=(W,E))
        self.hostname_entry.grid(column=1,columnspan=2, row=0, sticky=(W,E))

        self.username_label.grid(column=0, row=1, sticky=(W,E))
        self.username_entry.grid(column=1,columnspan=2, row=1, sticky=(W,E))

        self.password_label.grid(column=0, row=2, sticky=(W,E))
        self.password_entry.grid(column=1, columnspan=2,row=2, sticky=(W,E))
        
        self.upload_project_label.grid(column=0, row=3, sticky=(W,E))
        self.upload_project_entry.grid(column=1, columnspan=2,row=3, sticky=(W,E))
        self.upload_dataset_label.grid(column=0,row=4, pady=5, sticky=(W,E))
        self.dataset_name_entry.grid(column=1,row=4,columnspan=2, pady=5, sticky=(W,E))
        self.dataset_label.grid(column=0,row=5, pady=5, sticky=(W,E))
        if REMOTE:
            self.data_dir_option_menu.grid(column=0, row=6, columnspan=2, sticky=(W,E))
            self.button_update_list.grid(column=2,row=6, padx=6, sticky=(W,E))
            self.send_data_omero.grid(padx=(100, 10),columnspan=2,ipady=4,pady=5,row=7, )
       
        else :
            self.prediction_folder_label.grid(column=0, row=6,padx=5, sticky=(W,E))
            self.prediction_folder.grid(column=1,columnspan=2,row=6, sticky=(W,E))
        # grid config
        self.columnconfigure(0, weight=1)
        self.columnconfigure(1, weight=5)
        for i in range(6):
            self.rowconfigure(i, weight=1)

       
        # add send to omero button 
    def _update_pred_list(self):
        _,stdout,_ = REMOTE.exec_command('ls {}/data/pred'.format(MAIN_DIR))
        self.data_list = [e.replace('\n','') for e in stdout.readlines()]
        self.data_dir_option_menu.set_menu(self.data_list[0], *self.data_list)       
        popupmsg("Updated !")
        
        
    def send_to_omero(self): 
        # get the last folder modified/created
        _,stdout,stderr=REMOTE.exec_command("ls -td {}/data/pred/{}/*/ | head -1".format(MAIN_DIR, self.dataset_selected_omero() ))  
        last_folder = stdout.readline().replace('\n','')
        _,stdout,stderr=REMOTE.exec_command("source {}/bin/activate; cd {}; python -m biom3d.omero_uploader --username {} --password {} --hostname {} --project {} --path '{}' --dataset_name {}".format(VENV,MAIN_DIR, self.username_entry.get(), self.password_entry.get(), self.hostname_entry.get(), self.upload_project_entry.get(), last_folder,self.dataset_name_entry.get() ))
        
       
        while True: 
                    line = stdout.readline()
                    if not line:
                        break
                    if line:
                        print(line, end="")
        while True: # print error messages if needed
            line = stderr.readline()
            if not line:
                break
            if line:
                print(line, end="")
       
       
       
    def dataset_selected_omero(self, *args):
        global selected_dataset
        selected_dataset = self.data_dir.get()
        #print("Selected option:", selected_dataset)
        return selected_dataset  
class DownloadPrediction(ttk.LabelFrame):
    """
    REMOTE only! download output after prediction
    """
    def __init__(self, *arg, **kw):
        super(DownloadPrediction, self).__init__(*arg, **kw)

        assert REMOTE, "[Error] REMOTE must defined"

        self.input_folder_label = ttk.Label(self, text="Select a remote folder to download :")

        # define the dropdown menu
        _,stdout,_ = REMOTE.exec_command('ls {}/data/pred'.format(MAIN_DIR))
        self.data_list = [e.replace('\n','') for e in stdout.readlines()]
        if(len(self.data_list) == 0):
                self.data_dir = StringVar(value="Empty")
        else:   
            self.data_dir = StringVar(value=self.data_list[0])
        self.data_dir_option_menu = ttk.OptionMenu(self, self.data_dir, self.data_dir.get(), *self.data_list)
        self.button_update_list = ttk.Button(self, text="Update", command=self._update_pred_list)

        # or send the dataset to server
        self.get_data_label = ttk.Label(self, text="Select local folder to download into:")
        self.get_data_folder = FileDialog(self, mode='folder', textEntry="data/pred")
        self.get_data_button = ttk.Button(self,width=29,style="train_button.TLabel", text="Get data", command=self.get_data)

        self.input_folder_label.grid(column=0, row=0, columnspan=2, sticky=(W,E))
        self.data_dir_option_menu.grid(column=0, row=1, sticky=(W,E))
        self.button_update_list.grid(column=1, row=1,padx=5, sticky=(W,E))
        self.get_data_label.grid(column=0, row=2, columnspan=2, sticky=(W,E))
        self.get_data_folder.grid(column=0, row=3, columnspan=2, sticky=(W,E))
        self.get_data_button.grid(column=0, row=4, columnspan=2, pady=5,ipady=4, )

        self.columnconfigure(0, weight=10)
        self.columnconfigure(1, weight=1)
        for i in range(5):
            self.rowconfigure(i, weight=1)
    
    def get_data(self):
        # download dataset from the remote server to the local server
        
        # connect to remote
        ftp = REMOTE.open_sftp()

        # remote directory
        remotedir = "{}/data/pred/{}".format(MAIN_DIR, self.data_dir.get())

        # create local dir if it does not exist already
        localdir = os.path.join(self.get_data_folder.get(), self.data_dir.get())
        if not os.path.exists(localdir):
            os.makedirs(localdir, exist_ok=True)
        
        # copy files from remote to local
        ftp_get_folder(ftp, remotedir, localdir)
        popupmsg("Data sent to local !")
    def _update_pred_list(self):
        _,stdout,_ = REMOTE.exec_command('ls {}/data/pred'.format(MAIN_DIR))
        self.data_list = [e.replace('\n','') for e in stdout.readlines()]
        self.data_dir_option_menu.set_menu(self.data_list[0], *self.data_list)
        popupmsg("Prediction list updated !")

class PredictTab(ttk.Frame):
    def __init__(self, *arg, **kw):
        super(PredictTab, self).__init__(*arg, **kw)
        self.prediction_messages = ttk.Label(self, text="")
        self.use_omero_state = IntVar(value=0) 
        self.send_to_omero_state = IntVar(value=0) 
        # if platform=='linux' or REMOTE: # local Omero for linux only
        self.use_omero = ttk.Checkbutton(self, text="Use omero input directory", command=self.display_omero, variable=self.use_omero_state)
        self.input_dir = InputDirectory(self, text="Input directory", padding=[10,10,10,10])
        self.model_selection = ModelSelection(self, text="Model selection", padding=[10,10,10,10])
        self.send_to_omero = ttk.Checkbutton(self, text="Send Predictions to omero ", command=self.display_send_to_omero, variable=self.send_to_omero_state)
        if not REMOTE: self.output_dir = OutputDirectory(self, text="Output directory", padding=[10,10,10,10])
        self.button = ttk.Button(self, width=29,style="train_button.TLabel", text="Start", command=self.predict)
        if REMOTE: self.download_prediction = DownloadPrediction(self, text="Download predictions to local", padding=[10,10,10,10])

        # if platform=='linux' or REMOTE: # local Omero for linux only
        self.use_omero.grid(column=0,row=0,sticky=(W,E), pady=6)
        self.input_dir.grid(column=0,row=1,sticky=(W,E), pady=6)
        self.model_selection.grid(column=0,row=3,sticky=(W,E), pady=6)
        
        if not REMOTE: 
            self.output_dir.grid(column=0,row=5,sticky=(W,E), pady=6)
            self.send_to_omero.grid(column=0,row=4,sticky=(W,E), pady=6)

        self.button.grid(column=0,row=6,ipady=4, pady=4,padx=10,sticky=(S,N))
        if REMOTE: 
            self.download_prediction.grid(column=0, row=8, sticky=(W,E), pady=6)
            self.send_to_omero.grid(column=0,row=7,sticky=(W,E), pady=6)
    
        self.columnconfigure(0, weight=1)
        for i in range(7):
            self.rowconfigure(i, weight=1)
    
    def predict(self):
        # if use Omero then use Omero prediction
        if self.use_omero_state.get():
            obj="Dataset"+":"+self.omero_dataset.id.get()
         
            if REMOTE:
                
                # TODO: below, still OS dependant 
                _, stdout, stderr = REMOTE.exec_command("source {}/bin/activate; cd {}; python -m biom3d.omero_pred --obj {} --log {} --username {} --password {} --hostname {} ".format(VENV,
                    MAIN_DIR,
                    obj,
                    MAIN_DIR+'/logs/'+self.model_selection.logs_dir.get(), 
                    self.omero_connection.username.get(),
                    self.omero_connection.password.get(),
                    self.omero_connection.hostname.get()
                    ))
                while True: 
                    line = stdout.readline()
                    if not line:
                        break
                    if line:
                        print(line, end="")
                while True: # print error messages if needed
                    line = stderr.readline()
                    if not line:
                        break
                    if line:
                        print(line, end="")

                self.download_prediction._update_pred_list()
            else:
                target = "data/to_pred"
                if not os.path.isdir(target):
                    os.makedirs(target, exist_ok=True)
                print("Downloading Omero dataset into", target)
                p=biom3d.omero_pred.run(
                    obj=obj,
                    target=target,
                    log=self.model_selection.logs_dir.get(), 
                    dir_out=self.output_dir.data_dir.get(),
                    user=self.omero_connection.username.get(),
                    pwd=self.omero_connection.password.get(),
                    host=self.omero_connection.hostname.get()
                )           
                if self.send_to_omero_state.get():
                    biom3d.omero_uploader.run(username=self.send_to_omero_connection.username.get(),
                    password=self.send_to_omero_connection.password.get(),
                    hostname=self.send_to_omero_connection.hostname.get(),
                    project=int(self.send_to_omero_connection.upload_project_entry.get()),
                    dataset_name=self.send_to_omero_connection.dataset_name_entry.get(),
                    path=p)
        else: # if not use Omero
            if REMOTE:
                _, stdout, stderr = REMOTE.exec_command("source {}/bin/activate; cd {}; python -m biom3d.pred --log {} --dir_in {} --dir_out {}".format(VENV,
                    MAIN_DIR,
                    'logs/'+self.model_selection.logs_dir.get(), 
                    'data/to_pred/'+self.input_dir.data_dir.get(),
                    'data/pred/'+self.input_dir.data_dir.get(), # the default prediction output folder
                    ))
                while True: 
                    line = stdout.readline()
                    if not line:
                        break
                    if line:
                        print(line, end="")
                while True: # print error messages if needed
                    line = stderr.readline()
                    if not line:
                        break
                    if line:
                        print(line, end="")
                
                self.download_prediction._update_pred_list()
            else: 
                if self.send_to_omero_state.get():
                    target= self.send_to_omero_connection.prediction_folder.get()
                else :
                    target = self.output_dir.data_dir.get()
                self.prediction_messages.grid(column=0, row=6, columnspan=2, sticky=(W,E))
                self.prediction_messages.config(text="Prediction is running ...!")
                p = pred(
                    log=self.model_selection.logs_dir.get(),
                    dir_in=self.input_dir.data_dir.get(),
                    dir_out=target)
                self.prediction_messages.config(text="Prediction is Done !")
                if self.send_to_omero_state.get():
                    biom3d.omero_uploader.run(username=self.send_to_omero_connection.username.get(),
                    password=self.send_to_omero_connection.password.get(),
                    hostname=self.send_to_omero_connection.hostname.get(),
                    project=int(self.send_to_omero_connection.upload_project_entry.get()),
                    dataset_name=self.send_to_omero_connection.dataset_name_entry.get(),
                    path=p)
                popupmsg("Prediction done !")

    def display_omero(self):
        if self.use_omero_state.get():
            # hide the standard input dir and replace it by the Omero dir
            # hide
            self.input_dir.grid_remove()

            # place the new ones
            self.omero_connection = Connect2Omero(self, text="Connection to Omero server", padding=[10,10,10,10])
            self.omero_dataset = OmeroDataset(self, text="Selection of Omero dataset", padding=[10,10,10,10])

            self.omero_connection.grid(column=0,row=1,sticky=(W,E), pady=6)
            self.omero_dataset.grid(column=0,row=2,sticky=(W,E), pady=6)

        else:
            # hide omero 
            self.omero_connection.grid_remove()
            self.omero_dataset.grid_remove()

            # reset the input dir
            self.input_dir.grid(column=0,row=1,sticky=(W,E))
       
    def display_send_to_omero(self):
         if self.send_to_omero_state.get():
             # place the new ones
            self.send_to_omero_connection = OmeroUpload(self, text="Connection to Omero server", padding=[10,10,10,10])
            self.send_to_omero_connection.grid(column=0,row=9,sticky=(W,E), pady=6)
            
            if REMOTE :
                self.download_prediction.grid_remove()
                self.send_to_omero_connection.grid(column=0,row=8,sticky=(W,E), pady=6)
                
            else :
                self.output_dir.grid_remove()     
                self.send_to_omero_connection.grid(column=0,row=5,sticky=(W,E), pady=6)

         else:
            # hide omero 
            self.send_to_omero_connection.grid_remove()
            if REMOTE : self.download_prediction.grid(column=0, row=8, sticky=(W,E), pady=6)
            else: self.output_dir.grid(column=0,row=5,sticky=(W,E), pady=6)
          
#----------------------------------------------------------------------------
# Main loop

class Connect2Remote(ttk.LabelFrame):
    """
    Class to connect to remote server. use paramiko package
    """
    def __init__(self, *arg, **kw):
        super(Connect2Remote, self).__init__(*arg, **kw)

        # widgets definitions
        self.hostname_label = ttk.Label(self, text='Server address:')
        self.hostname = StringVar(value="")
        self.hostname_entry = ttk.Entry(self, textvariable=self.hostname)

        self.username_label = ttk.Label(self, text='User name:')
        self.username = StringVar(value="biome")
        self.username_entry = ttk.Entry(self, textvariable=self.username)

        self.password_label = ttk.Label(self, text='Password:')
        self.password = StringVar(value="")
        self.password_entry = ttk.Entry(self, textvariable=self.password, show='*')
        
        self.main_dir_label = ttk.Label(self, text='Folder of Biom3d repository on remote server:')
        self.main_dir = StringVar(value="/home/biome/biom3d")
        self.main_dir_entry = ttk.Entry(self, textvariable=self.main_dir)

        self.venv_label = ttk.Label(self, text='(Optional) Name of the virtual environement on remote server:')
        self.venv_name = StringVar(value="")
        self.venv_entry = ttk.Entry(self, textvariable=self.venv_name)
        
        self.use_proxy_state = IntVar() 
        self.use_proxy = ttk.Checkbutton(self, text="Use proxy server for ssh connexion", command=self.display_proxy, variable=self.use_proxy_state)
        
        # self.use_proxy.state(['!alternate'])

        # place widgets
        self.hostname_label.grid(column=0, row=0, sticky=(W,E))
        self.hostname_entry.grid(column=1, row=0, sticky=(W,E))

        self.username_label.grid(column=0, row=1, sticky=(W,E))
        self.username_entry.grid(column=1, row=1, sticky=(W,E))

        self.password_label.grid(column=0, row=2, sticky=(W,E))
        self.password_entry.grid(column=1, row=2, sticky=(W,E))

        self.main_dir_label.grid(column=0, row=3, sticky=(W,E))
        self.main_dir_entry.grid(column=1, row=3, sticky=(W,E))

        self.venv_label.grid(column=0, row=4, sticky=(W,E))
        self.venv_entry.grid(column=1, row=4, sticky=(W,E))
        
        self.use_proxy.grid(column=0, columnspan=2, row=5, sticky=(W))

        # grid config
        self.columnconfigure(0, weight=1)
        self.columnconfigure(1, weight=5)
        for i in range(5):
            self.rowconfigure(i, weight=1)
    
    def display_proxy(self):
        """
        print the proxy configuration widgets
        """
        if self.use_proxy_state.get(): # place widgets
            self.proxy_hostname_label = ttk.Label(self, text='Proxy server address:')
            self.proxy_hostname = StringVar(value="")
            self.proxy_hostname_entry = ttk.Entry(self, textvariable=self.proxy_hostname)

            self.proxy_username_label = ttk.Label(self, text='Proxy user name:')
            self.proxy_username = StringVar(value="")
            self.proxy_username_entry = ttk.Entry(self, textvariable=self.proxy_username)

            self.proxy_password_label = ttk.Label(self, text='Proxy password:')
            self.proxy_password = StringVar(value="")
            self.proxy_password_entry = ttk.Entry(self, textvariable=self.proxy_password, show='*')


            self.proxy_hostname_label.grid(column=0, row=5, sticky=(W,E))
            self.proxy_hostname_entry.grid(column=1, row=5, sticky=(W,E))

            self.proxy_username_label.grid(column=0, row=6, sticky=(W,E))
            self.proxy_username_entry.grid(column=1, row=6, sticky=(W,E))

            self.proxy_password_label.grid(column=0, row=7, sticky=(W,E))
            self.proxy_password_entry.grid(column=1, row=7, sticky=(W,E))

            for i in range(5,8):
                self.rowconfigure(i, weight=1)
        else: # remove the widget
            self.proxy_hostname_label.grid_remove()
            self.proxy_hostname_entry.grid_remove()

            self.proxy_username_label.grid_remove()
            self.proxy_username_entry.grid_remove()

            self.proxy_password_label.grid_remove()
            self.proxy_password_entry.grid_remove()



class Root(Tk):
    def __init__(self):
        # Stage 0 (root)
        super(Root, self).__init__()

        # title
        self.title("Biom3d")

        # windows dimension and positioning
        window_width = 725
        window_height = 785

        ## get the screen dimension
        global screen_width
        screen_width = self.winfo_screenwidth()
        screen_height = self.winfo_screenheight()

        ## find the center point
        center_x = int(screen_width/2 - window_width / 2)
        center_y = int(screen_height/2 - window_height / 2)

        ## set the position of the window to the center of the screen
        self.geometry(f'{window_width}x{window_height}+{center_x}+{center_y}')

        self.minsize(360,360)
        # self.iconbitmap("biom3d/microscope.png")
        self.config(background='#D00000')

        # Initiate styles
        init_styles()

        # background

        self.background = ttk.Frame(self, padding=PADDING, style='red.TFrame')
        self.background.grid(column=0, row=0, sticky=(N, W, E, S))
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)
        # Stage 1.1 (root --> local of server)

        self.local_or_remote = ttk.Frame(self.background, padding=[20,20,20,20])
        self.local_or_remote.pack(expand=YES, fill='x', padx=20, pady=20)

        ## Stage 2 (local_or_remote --> frame)

    
        style = ttk.Style()
        style.configure("BW.TLabel",background = '#CD5C5C', foreground = 'white', font=('Helvetica', 9), width = 18, borderwidth=3, focusthickness=7, relief='raised', focuscolor='none', anchor='c', height= 15)
        self.title_label = ttk.Label(self.local_or_remote, text="Biom3d", font=("Montserrat", 18))
        self.welcome_message = ttk.Label(self.local_or_remote, text="Welcome!\n\nBiom3d is an easy-to-use tool to train and use deep learning models for segmenting three dimensional images. You can either start locally, if your computer has a good graphic card (NVIDIA Geforce RTX 1080 or higher) or connect remotelly on a computer with such a graphic card.\n\nIf you need help, check our GitHub repository here: https://github.com/GuillaumeMougeot/biom3d", anchor="w", justify=LEFT, wraplength=640)

        self.start_locally = ttk.Button(self.local_or_remote, text="Start locally", style='BW.TLabel', command=lambda: self.main(remote=False))

        self.start_remotelly_frame = Connect2Remote(self.local_or_remote, text="Connect to remote server", padding=[10,10,10,10])
        self.start_remotelly_button = ttk.Button(self.local_or_remote, text='Start remotelly',style='BW.TLabel', command=lambda: self.main(remote=True))

        self.title_label.grid(column=0, row=0, sticky=W)
        self.welcome_message.grid(column=0, row=1, sticky=(W,E), pady=12)
        
        modulename='biom3d'
        if modulename in sys.modules :
            self.start_locally.grid(column=0, row=2, ipady=4, pady=12)

        self.start_remotelly_frame.grid(column=0, row=3, sticky=(W,E), pady=12)
        self.start_remotelly_button.grid(column=0, row=4, ipady=4, pady=5)

        # grid config
        self.local_or_remote.columnconfigure(0, weight=1)
        for i in range(5):
            self.local_or_remote.rowconfigure(i, weight=1)

        # setup client for remote access
        self.client = None
    
    def main(self, remote=False):

        # remote access connections
        if remote:
            global REMOTE
            global MAIN_DIR

            print('Connecting to server...')
            

            # connection
            # if use proxy server then connect to proxy first
            if self.start_remotelly_frame.use_proxy_state.get():
                proxy_server = paramiko.SSHClient()
                proxy_server.set_missing_host_key_policy(paramiko.AutoAddPolicy())
                proxy_server.connect(
                    hostname=self.start_remotelly_frame.proxy_hostname.get(),
                    username=self.start_remotelly_frame.proxy_username.get(),
                    password=self.start_remotelly_frame.proxy_password.get())
                io_tupple = proxy_server.exec_command('nc {} 22'.format(self.start_remotelly_frame.hostname.get()))

                proxy = ParaProxy(*io_tupple)


                REMOTE=paramiko.SSHClient()
                REMOTE.set_missing_host_key_policy(paramiko.AutoAddPolicy())
                REMOTE.connect(
                    hostname=self.start_remotelly_frame.hostname.get(),
                    username=self.start_remotelly_frame.username.get(),
                    password=self.start_remotelly_frame.password.get(),
                    sock=proxy) # the socket parameter is the proxy server

            else: 
                REMOTE=paramiko.SSHClient()
                REMOTE.set_missing_host_key_policy(paramiko.AutoAddPolicy())
                REMOTE.connect(
                    hostname=self.start_remotelly_frame.hostname.get(),
                    username=self.start_remotelly_frame.username.get(),
                    password=self.start_remotelly_frame.password.get())

            MAIN_DIR = self.start_remotelly_frame.main_dir.get()
            
            # Specify path
            path1 = self.start_remotelly_frame.venv_entry.get()
            path = "'"+path1+"'"
            
            # Check whether the specified
            # path exists or not
            _,stdout,stderr = REMOTE.exec_command('python -c "from pathlib import Path; obj = Path({}) ;print(obj.exists()) "  '.format(path)) 
            
            while True: 
                    line = stdout.readline()
                    if not line:
                        break
                    if line:
                        print(line, end="")
            while True: # print error messages if needed
                line = stderr.readline()
                if not line:
                    break
                if line:
                    print(line, end="")
            # path_to_venv = path1.split("/")  
            global VENV 
            # VENV = path_to_venv[-1]
            VENV = path
            print("virtual environment name: ",VENV)
            
        modulename='biom3d'
        if modulename not in sys.modules and not REMOTE :
            popupmsg(" you can't access local interface in international area please contact the national space agency")
        else :  
            # Stage 1.2 (root -> root_frame)
            self.root_frame = ttk.Frame(self, padding=PADDING, style='red.TFrame')
            self.root_frame.grid(column=0, row=0, sticky=(N, W, E, S))
            self.columnconfigure(0, weight=1)
            self.rowconfigure(0, weight=1)

            # Stage 2.1  (root_frame -> notebook)
            self.tab_parent = ttk.Notebook(self.root_frame, style='red.TNotebook')
            self.tab_parent.pack(expand=YES, fill='both', padx=6, pady=6)

            # Stage 3 (notebook -> preprocess - train - predict - omero)
            
            self.train_tab = ttk.Frame(self.tab_parent, padding=PADDING)
            self.predict_tab = ttk.Frame(self.tab_parent, padding=PADDING)
            
            self.tab_parent.add(self.train_tab, text="Preprocess & Train")
            self.tab_parent.add(self.predict_tab, text="Predict")
     
            # Stage 4 (predict_tab -> predict_tab_frame)
            self.predict_tab_frame = PredictTab(self.predict_tab)
            self.predict_tab_frame.grid(column=0, row=0, sticky=(N,W,E), pady=24, padx=12)
            self.predict_tab.columnconfigure(0, weight=1)
            self.predict_tab.rowconfigure(0, weight=1)

            # Stage 4 (train_tab -> train_tab_frame)
            self.train_tab_frame = TrainTab(master=self.train_tab)
            self.train_tab_frame.grid(column=0, row=0, sticky=(N,W,E), pady=24, padx=12)
            self.train_tab.columnconfigure(0, weight=1)
            self.train_tab.rowconfigure(0, weight=1)
            
            #Home button
            
            self.home_button_train=ttk.Button(self.train_tab,text="Home",command=self.refresh) #TODO : refresh doesnt work properly !
            self.home_button_pred=ttk.Button(self.predict_tab,text="Home",command=self.refresh)
            #self.home_button_train.grid(column=0, row=0, sticky=(S,W)) 
            #self.home_button_pred.grid(column=0, row=0, sticky=(S,W))
    def refresh(self):
            self.destroy()
            self.__init__()
            
            
if __name__=='__main__':
    
    root = Root()

    try: # avoid blury UI on Windows
        if platform=='win32':
            from ctypes import windll
            windll.shcore.SetProcessDpiAwareness(1)
    finally:
        root.mainloop()

    