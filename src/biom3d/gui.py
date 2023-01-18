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
from tkinter import LEFT, ttk, Tk, N, W, E, S, YES, IntVar, StringVar
from tkinter import filedialog
import paramiko
from stat import S_ISDIR, S_ISREG # for recursive download
import os 
import yaml
from sys import platform 
# if platform=='linux': # only import if linux because windows omero plugin requires Visual Studio Install which is too big
import argparse

from biom3d.configs.unet_default import CONFIG
from biom3d.preprocess import preprocess
from biom3d.auto_config import auto_config

# the packages below are only needed for the local version of the GUI
# WARNING! the lines below must be commented when deploying the remote version,
# and uncommented when installing the local version.
from biom3d.pred import pred
from biom3d.builder import Builder
# import omero_pred

#----------------------------------------------------------------------------
# Constants 
# remote or local

REMOTE = False

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
    # theme_style.configure('TFrame', paddind=)

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
# general utils (also in biom3d.utils)

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
        self.button = ttk.Button(self,text="Browse", command=self.command)
        self.button.grid(column=2,row=1, sticky=(W))

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
# Preprocess tab

class PreprocessFolderSelection(ttk.LabelFrame):
    def __init__(self, *arg, **kw):
        super(PreprocessFolderSelection, self).__init__(*arg, **kw)

        # Define elements
        ## image folder
        self.label1 = ttk.Label(self, text="Select the folder containing the raw images:", anchor="sw", background='white')
        self.img_dir = FileDialog(self, mode='folder', textEntry='D:/code/python/3dnucleus/data/img')

        ## mask folder
        self.label2 = ttk.Label(self, text="Select a folder containing the annotated masks:", anchor="sw", background='white')
        self.msk_dir = FileDialog(self, mode='folder', textEntry="D:/code/python/3dnucleus/data/msk")

        ## number of classes
        self.label3 = ttk.Label(self, text="Enter the number of classes:", anchor="sw", background='white')
        self.num_classes = IntVar(value=1)
        self.classes = ttk.Entry(self, textvariable=self.num_classes)

        ## Output folders for images
        self.label4 = ttk.Label(self, text="(Optional) Select a folder to store the new preprocessed images:", anchor="sw", background='white')
        self.img_outdir = FileDialog(self, mode='folder', textEntry="")
    
        ## Output folders for masks
        self.label5 = ttk.Label(self, text="(Optional) Select a folder to store the new preprocessed masks:", anchor="sw", background='white')
        self.msk_outdir = FileDialog(self, mode='folder', textEntry="")

        # Position elements
        self.label1.grid(column=0, row=0, sticky=W)
        self.img_dir.grid(column=0, row=1, sticky=(W,E))

        self.label2.grid(column=0,row=2, sticky=W)
        self.msk_dir.grid(column=0,row=3, sticky=(W,E))

        self.label3.grid(column=0,row=4, sticky=W)
        self.classes.grid(column=0,row=5, sticky=(W,E))

        self.label4.grid(column=0,row=6, sticky=W)
        self.img_outdir.grid(column=0,row=7, sticky=(W,E))

        self.label5.grid(column=0,row=8, sticky=W)
        self.msk_outdir.grid(column=0,row=9, sticky=(W,E))
        
        # Configure columns
        self.columnconfigure(0, weight=1)

        for i in range(10):
            self.rowconfigure(i, weight=1)

class PreprocessTab(ttk.Frame):
    def __init__(self, *arg, **kw):
        super(PreprocessTab, self).__init__(*arg, **kw)

        # widget definition
        self.folder_selection = PreprocessFolderSelection(self, text="Local folder path configurations", padding=[10,10,10,10])
        self.button = ttk.Button(self, text="Start", command=self.preprocess)
        self.done_label = ttk.Label(self, text="", anchor="sw", background='white')

        ## send dataset to remote server
        if REMOTE:
            self.send_data_label = ttk.Label(self, text="To send the preprocessed dataset on the remote server: 1. define a unique name below, 2. press the button")
            self.send_data_name = StringVar(value="nucleus_0001")
            self.send_data_entry = ttk.Entry(self, textvariable=self.send_data_name)
            self.send_data_button = ttk.Button(self, text="Send data to remote server", command=self.send_data)
            self.send_data_finish = ttk.Label(self, text="")

        # widget placement
        self.folder_selection.grid(column=0,row=0,sticky=(N,W,E), pady=3)
        self.button.grid(column=0,row=1,sticky=(N,W,E))
        self.done_label.grid(column=0,row=2,sticky=(N,W,E))

        if REMOTE:
            self.send_data_label.grid(column=0, row=3, sticky=(W,E), pady=10)
            self.send_data_entry.grid(column=0, row=4, sticky=(W,E))
            self.send_data_button.grid(column=0, row=5, sticky=(W,E), pady=3)
            self.send_data_finish.grid(column=0, row=6, sticky=(W,E), pady=3)
    
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)

        if REMOTE:
            for i in range(2,7):
                self.rowconfigure(i, weight=1)
    
    def preprocess(self):
        # set automatically the output directories if empty
        if self.folder_selection.img_outdir.get()=="":
            self.folder_selection.img_outdir.set(self.folder_selection.img_dir.get()+'_out')
        if self.folder_selection.msk_outdir.get()=="":
            self.folder_selection.msk_outdir.set(self.folder_selection.msk_dir.get()+'_out')

        preprocess(
            img_dir=self.folder_selection.img_dir.get(),
            msk_dir=self.folder_selection.msk_dir.get(),
            img_outdir=self.folder_selection.img_outdir.get(),
            msk_outdir=self.folder_selection.msk_outdir.get(),
            num_classes=self.folder_selection.num_classes.get()+1,
            remove_bg=False)
        if REMOTE:
            done_label_text = "Done preprocessing! You can send your dataset to the server before training."
        else:
            done_label_text = "Done preprocessing! You can start training."
        self.done_label.config(text=done_label_text)
    
    def send_data(self):
        ftp = REMOTE.open_sftp()

        # copy folders 
        remote_dir_img = "{}/data/{}/img_out".format(MAIN_DIR,self.send_data_name.get())
        remote_dir_msk = "{}/data/{}/msk_out".format(MAIN_DIR,self.send_data_name.get())
        ftp_put_folder(ftp, localpath=self.folder_selection.img_outdir.get(), remotepath=remote_dir_img)
        ftp_put_folder(ftp, localpath=self.folder_selection.msk_outdir.get(), remotepath=remote_dir_msk)

        self.send_data_finish.config(text="Data sent!")

#----------------------------------------------------------------------------
# train tab

class TrainFolderSelection(ttk.LabelFrame):
    def __init__(self, preprocess_tab=None, *arg, **kw):
        super(TrainFolderSelection, self).__init__(*arg, **kw)

        # Define elements
        # use preprocessing values
        self.use_preprocessing_button = ttk.Button(self, text="Use preprocessing values", command=self.use_preprocessing)
        self.preprocess_tab = preprocess_tab

        ## image folder
        self.label1 = ttk.Label(self, text="Select the folder containing the preprocessed images:", anchor="sw", background='white')
        ## mask folder
        self.label2 = ttk.Label(self, text="Select a folder containing the preprocessed masks:", anchor="sw", background='white')

        if REMOTE:
            # get dataset list
            _,stdout,_ = REMOTE.exec_command('ls {}/data'.format(MAIN_DIR))
            self.data_list = [e.replace('\n','') for e in stdout.readlines()]

            # define the dropdown menu
            self.data_dir = StringVar(value=self.data_list[0])
            self.data_dir_option_menu = ttk.OptionMenu(self, self.data_dir, self.data_list[0], *self.data_list)
            self.img_outdir = StringVar("")
            self.msk_outdir = StringVar("")
            self._update_data_dir()

        else:
            self.img_outdir = FileDialog(self, mode='folder', textEntry='D:/code/python/3dnucleus/data/img_out')        
            self.msk_outdir = FileDialog(self, mode='folder', textEntry="D:/code/python/3dnucleus/data/msk_out")

        ## number of classes
        self.label3 = ttk.Label(self, text="Enter the number of classes:", anchor="sw", background='white')
        self.num_classes = IntVar(value=1)
        self.classes = ttk.Entry(self, textvariable=self.num_classes)

        # Position elements
        self.use_preprocessing_button.grid(column=0, row=0, sticky=(W,E))

        self.label1.grid(column=0, row=1, sticky=W)
        
        if REMOTE:
            self.data_dir_option_menu.grid(column=0, row=2, sticky=(W,E))
        else:
            self.img_outdir.grid(column=0, row=2, sticky=(W,E))
            self.label2.grid(column=0,row=3, sticky=W)
            self.msk_outdir.grid(column=0,row=4, sticky=(W,E))

        self.label3.grid(column=0,row=5, sticky=W)
        self.classes.grid(column=0,row=6, sticky=(W,E))

        
        # Configure columns
        self.columnconfigure(0, weight=1)

        for i in range(7):
            self.rowconfigure(i, weight=1)
    
    def _update_data_dir(self):
        """
        update the names of image and mask directories
        """
        base_name = "{}/data/{}/".format(MAIN_DIR, self.data_dir.get())
        self.img_outdir.set(base_name + "img_out")
        self.msk_outdir.set(base_name + "msk_out")

    def use_preprocessing(self):
        if REMOTE:
            # use remote dirs names
            
            # update the list of data directories
            _,stdout,_ = REMOTE.exec_command('ls {}/data'.format(MAIN_DIR))
            self.data_list = [e.replace('\n','') for e in stdout.readlines()]

            # update option menu list 
            self.data_dir_option_menu.set_menu(self.data_list[0], *self.data_list)
            
            self.data_dir.set(self.preprocess_tab.send_data_name.get())
            self._update_data_dir()
        else:
            self.img_outdir.set(self.preprocess_tab.folder_selection.img_outdir.get())
            self.msk_outdir.set(self.preprocess_tab.folder_selection.msk_outdir.get())

        self.num_classes.set(self.preprocess_tab.folder_selection.num_classes.get())

class ConfigFrame(ttk.LabelFrame):
    def __init__(self, train_folder_selection=None, *arg, **kw):
        super(ConfigFrame, self).__init__(*arg, **kw)

        # widgets definitions
        self.auto_config_button = ttk.Button(self, text="Auto-configuration", command=self.auto_config)
        self.img_outdir = train_folder_selection.img_outdir
        self.auto_config_finished = ttk.Label(self, text="")

        self.num_epochs_label = ttk.Label(self, text='Number of epochs:')
        self.num_epochs = IntVar(value=10)
        self.num_epochs_entry = ttk.Entry(self, textvariable=self.num_epochs)

        self.batch_size_label = ttk.Label(self, text='Batch size (int):')
        self.batch_size = IntVar(value=2)
        self.batch_size_entry = ttk.Entry(self, textvariable=self.batch_size)

        self.patch_size_label = ttk.Label(self, text='Patch size ([int int int]):')
        self.patch_size = StringVar(value="[128 128 128]")
        self.patch_size_entry = ttk.Entry(self, textvariable=self.patch_size)

        self.aug_patch_size_label = ttk.Label(self, text='Augmentation patch size ([int int int]):')
        self.aug_patch_size = StringVar(value="[160 160 160]")
        self.aug_patch_size_entry = ttk.Entry(self, textvariable=self.aug_patch_size)

        self.num_pools_label = ttk.Label(self, text='Number of pool in the U-Net model ([int int int]):')
        self.num_pools = StringVar(value="[5 5 5]")
        self.num_pools_entry = ttk.Entry(self, textvariable=self.num_pools)

        # place widgets
        self.auto_config_button.grid(column=0, row=0, columnspan=2, sticky=(W,E))
        self.auto_config_finished.grid(column=0, row=1, columnspan=2, sticky=(W,E))

        self.num_epochs_label.grid(column=0, row=2, sticky=(W,E))
        self.num_epochs_entry.grid(column=1, row=2, sticky=W)

        self.batch_size_label.grid(column=0, row=3, sticky=(W,E))
        self.batch_size_entry.grid(column=1, row=3, sticky=W)
        
        self.patch_size_label.grid(column=0, row=4, sticky=(W,E))
        self.patch_size_entry.grid(column=1, row=4, sticky=W)

        self.aug_patch_size_label.grid(column=0, row=5, sticky=(W,E))
        self.aug_patch_size_entry.grid(column=1, row=5, sticky=W)

        self.num_pools_label.grid(column=0, row=6, sticky=(W,E))
        self.num_pools_entry.grid(column=1, row=6, sticky=W)

        # grid config
        self.columnconfigure(0, weight=1)
        self.columnconfigure(1, weight=10)
        for i in range(6):
            self.rowconfigure(i, weight=1)

    def auto_config(self):
        self.auto_config_finished.config(text="Auto-configuration, please wait...")

        if REMOTE:
            _,stdout,stderr=REMOTE.exec_command("cd {}; python -m biom3d.auto_config --img_dir {} --min_dis".format(MAIN_DIR, self.img_outdir.get()))
            auto_config_results = stdout.readlines()
            auto_config_results = [e.replace('\n','') for e in auto_config_results]
            
            # error management
            if len(auto_config_results)!=4:
                print("[Error] Auto-config error:", auto_config_results)
            while True:
                line = stderr.readline()
                if not line:
                    break
                print(line, end="")

            batch, aug_patch, patch, pool = auto_config_results
        else: 
            batch, aug_patch, patch, pool = auto_config(self.img_outdir.get())

        self.batch_size.set(batch)
        self.aug_patch_size.set(aug_patch)
        self.patch_size.set(patch)
        self.num_pools.set(pool)

        self.auto_config_finished.config(text="Auto-configuration done!")

class TrainTab(ttk.Frame):
    def __init__(self, preprocess_tab=None, *arg, **kw):
        super(TrainTab, self).__init__(*arg, **kw)

        self.folder_selection = TrainFolderSelection(preprocess_tab=preprocess_tab, master=self, text="Folder path configurations", padding=[10,10,10,10])
        self.config_selection = ConfigFrame(train_folder_selection=self.folder_selection, master=self, text="Training configuration", padding=[10,10,10,10])

        self.builder_name_label = ttk.Label(self, text="Set a name for the builder folder (folder containing your future model):")
        self.builder_name = StringVar(value="unet_example")
        self.builder_name_entry = ttk.Entry(self, textvariable=self.builder_name)
        self.train_button = ttk.Button(self, text="Start", command=self.train)
        self.train_done = ttk.Label(self, text="")

        # set default values of train folders with the ones used for preprocess tab
        if REMOTE:
            self.folder_selection.data_dir.set(preprocess_tab.send_data_name.get())
        else: 
            self.folder_selection.img_outdir.set(preprocess_tab.folder_selection.img_outdir.get())
            self.folder_selection.msk_outdir.set(preprocess_tab.folder_selection.msk_outdir.get())
        self.folder_selection.num_classes.set(preprocess_tab.folder_selection.num_classes.get())

        self.folder_selection.grid(column=0,row=0,sticky=(N,W,E), pady=3)
        self.config_selection.grid(column=0,row=1,sticky=(N,W,E), pady=3)
        self.builder_name_label.grid(column=0, row=2, sticky=(W,E), pady=3)
        self.builder_name_entry.grid(column=0, row=3, sticky=(W,E))
        self.train_button.grid(column=0, row=4, sticky=(W,E))
        self.train_done.grid(column=0, row=5, sticky=W)

    
        self.columnconfigure(0, weight=1)
        for i in range(6):
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
        self.train_done.config(text="Training, please wait...")

        cfg = CONFIG

        # set the configuration
        cfg.IMG_DIR = self.folder_selection.img_outdir.get()
        cfg = nested_dict_change_value(cfg, 'img_dir', cfg.IMG_DIR)

        cfg.MSK_DIR = self.folder_selection.msk_outdir.get()
        cfg = nested_dict_change_value(cfg, 'msk_dir', cfg.MSK_DIR)

        cfg.DESC = self.builder_name.get()

        cfg.NUM_CLASSES = self.folder_selection.num_classes.get()
        cfg = nested_dict_change_value(cfg, 'num_classes', cfg.NUM_CLASSES)

        cfg.NB_EPOCHS = self.config_selection.num_epochs.get()

        cfg.BATCH_SIZE = self.config_selection.batch_size.get()
        cfg = nested_dict_change_value(cfg, 'batch_size', cfg.BATCH_SIZE)

        cfg.PATCH_SIZE = self.str2list(self.config_selection.patch_size.get())
        cfg = nested_dict_change_value(cfg, 'patch_size', cfg.PATCH_SIZE)

        cfg.AUG_PATCH_SIZE = self.str2list(self.config_selection.aug_patch_size.get())
        cfg = nested_dict_change_value(cfg, 'aug_patch_size', cfg.AUG_PATCH_SIZE)

        cfg.NUM_POOLS = self.str2list(self.config_selection.num_pools.get())
        cfg = nested_dict_change_value(cfg, 'num_pools', cfg.NUM_POOLS)

        if REMOTE:
            # if remote store the config file in a temp file
            save_config("config.yaml", cfg)
            # copy it
            ftp = REMOTE.open_sftp()
            ftp.put("config.yaml", MAIN_DIR+"/config.yaml")
            ftp.close()
            # delete the temp file
            os.remove("config.yaml")

             # run the training and store the output in an output file 
            # https://askubuntu.com/questions/1336685/how-do-i-save-to-a-file-and-simultaneously-view-terminal-output 
            _,stdout,stderr=REMOTE.exec_command("cd {}; python -m biom3d.train --config_yaml config.yaml | tee log.out".format(MAIN_DIR))
            
            # print the stdout continuously
            # from https://stackoverflow.com/questions/55642555/real-time-output-for-paramiko-exec-command  
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

                # TODO: copy the event file to local

        else:
            # run the training
            builder = Builder(config=cfg,path=None)
            builder.run_training()

        self.train_done.config(text="Training done!")


        

#----------------------------------------------------------------------------
# Precition tab

class InputDirectory(ttk.LabelFrame):
    def __init__(self, *arg, **kw):
        super(InputDirectory, self).__init__(*arg, **kw)

        self.input_folder_label = ttk.Label(self, text="Select a folder containing images for prediction:")
        self.input_folder_label.grid(column=0, row=0, sticky=(W,E))

        if REMOTE: 
            # if remote, print the list of available dataset or offer the option to send a local one on the server.

            # define the dropdown menu
            _,stdout,_ = REMOTE.exec_command('ls {}/data/to_pred'.format(MAIN_DIR))
            self.data_list = [e.replace('\n','') for e in stdout.readlines()]
            self.data_dir = StringVar(value=self.data_list[0])
            self.data_dir_option_menu = ttk.OptionMenu(self, self.data_dir, self.data_list[0], *self.data_list)

            # or send the dataset to server
            self.send_data_label = ttk.Label(self, text="Or send a new dataset of raw images to the server:")
            self.send_data_folder = FileDialog(self, mode='folder', textEntry="data/to_pred")
            self.send_data_button = ttk.Button(self, text="Send data", command=self.send_data)


            self.data_dir_option_menu.grid(column=0, row=1, sticky=(W,E))
            self.send_data_label.grid(column=0, row=2, sticky=(W,E))
            self.send_data_folder.grid(column=0, row=3, sticky=(W,E))
            self.send_data_button.grid(column=0, row=4, sticky=(W,E))

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

class Connect2Omero(ttk.LabelFrame):
    def __init__(self, *arg, **kw):
        super(Connect2Omero, self).__init__(*arg, **kw)

        # widgets definitions
        self.hostname_label = ttk.Label(self, text='Omero server address:')
        self.hostname = StringVar(value="omero.igred.fr")
        self.hostname_entry = ttk.Entry(self, textvariable=self.hostname)

        self.username_label = ttk.Label(self, text='User name:')
        self.username = StringVar(value="biome")
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

        self.option_list = ["Dataset", "Project"]
        self.option = StringVar(value=self.option_list[0])
        self.option_menu = ttk.OptionMenu(self, self.option, self.option_list[0], *self.option_list)

        self.label_id = ttk.Label(self, text="ID:")
        self.id = StringVar(value="22")
        self.id_entry = ttk.Entry(self, textvariable=self.id)

        self.option_menu.grid(column=0, row=0, sticky=(W,E))
        self.label_id.grid(column=1, row=0, sticky=(E))
        self.id_entry.grid(column=2, row=0, sticky=(W,E))

        self.columnconfigure(0, weight=1)
        self.columnconfigure(1, weight=1)
        self.columnconfigure(2, weight=1)
        self.rowconfigure(0, weight=1)

class ModelSelection(ttk.LabelFrame):
    def __init__(self, *arg, **kw):
        super(ModelSelection, self).__init__(*arg, **kw)

        # Define elements
        # get model list
        if REMOTE:
            _,stdout,_ = REMOTE.exec_command('ls {}/logs'.format(MAIN_DIR))
            self.logs_list = [e.replace('\n','') for e in stdout.readlines()]

            # define the dropdown menu
            self.logs_dir = StringVar(value=self.logs_list[0])
            self.logs_dir_option_menu = ttk.OptionMenu(self, self.logs_dir, self.logs_list[0], *self.logs_list)
            self.button_update_list = ttk.Button(self, text="Update", command=self._update_logs_list)

            self.logs_dir_option_menu.grid(column=0, row=0, sticky=(W,E))
            self.button_update_list.grid(column=1, row=0, sticky=(W,E))

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

class OutputDirectory(ttk.LabelFrame):
    def __init__(self, *arg, **kw):
        super(OutputDirectory, self).__init__(*arg, **kw)

        if REMOTE: 
            # if remote, only print an information message indicating where to find the prediction folder on the server.
            # self.default_label = ttk.Label(self, text="Impossible to change the default folders when using remote access. Default downloading folder is 'data/to_pred'; default predictions folder is 'data/pred'")
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
        self.data_dir = StringVar(value=self.data_list[0])
        self.data_dir_option_menu = ttk.OptionMenu(self, self.data_dir, self.data_list[0], *self.data_list)
        self.button_update_list = ttk.Button(self, text="Update", command=self._update_pred_list)

        # or send the dataset to server
        self.get_data_label = ttk.Label(self, text="Select local folder to download into:")
        self.get_data_folder = FileDialog(self, mode='folder', textEntry="data/pred")
        self.get_data_button = ttk.Button(self, text="Get data", command=self.get_data)

        self.input_folder_label.grid(column=0, row=0, columnspan=2, sticky=(W,E))
        self.data_dir_option_menu.grid(column=0, row=1, sticky=(W,E))
        self.button_update_list.grid(column=1, row=1, sticky=(W,E))
        self.get_data_label.grid(column=0, row=2, columnspan=2, sticky=(W,E))
        self.get_data_folder.grid(column=0, row=3, columnspan=2, sticky=(W,E))
        self.get_data_button.grid(column=0, row=4, columnspan=2, sticky=(W,E))

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
    
    def _update_pred_list(self):
        _,stdout,_ = REMOTE.exec_command('ls {}/data/pred'.format(MAIN_DIR))
        self.data_list = [e.replace('\n','') for e in stdout.readlines()]
        self.data_dir_option_menu.set_menu(self.data_list[0], *self.data_list)
        

class PredictTab(ttk.Frame):
    def __init__(self, *arg, **kw):
        super(PredictTab, self).__init__(*arg, **kw)

        self.use_omero_state = IntVar(value=0) 
        # if platform=='linux' or REMOTE: # local Omero for linux only
        self.use_omero = ttk.Checkbutton(self, text="Use omero input directory", command=self.display_omero, variable=self.use_omero_state)
        self.input_dir = InputDirectory(self, text="Input directory", padding=[10,10,10,10])
        self.model_selection = ModelSelection(self, text="Model selection", padding=[10,10,10,10])
        if not REMOTE: self.output_dir = OutputDirectory(self, text="Output directory", padding=[10,10,10,10])
        self.button = ttk.Button(self, text="Start", command=self.predict)
        if REMOTE: self.download_prediction = DownloadPrediction(self, text="Download predictions to local", padding=[10,10,10,10])

        # if platform=='linux' or REMOTE: # local Omero for linux only
        self.use_omero.grid(column=0,row=0,sticky=(W,E), pady=6)
        self.input_dir.grid(column=0,row=1,sticky=(W,E), pady=6)
        self.model_selection.grid(column=0,row=3,sticky=(W,E), pady=6)
        if not REMOTE: self.output_dir.grid(column=0,row=4,sticky=(W,E), pady=6)
        self.button.grid(column=0,row=5,sticky=(W,E), pady=6)
        if REMOTE: self.download_prediction.grid(column=0, row=6, sticky=(W,E), pady=6)
    
        self.columnconfigure(0, weight=1)
        for i in range(7):
            self.rowconfigure(i, weight=1)
    
    def predict(self):
        # if use Omero then use Omero prediction
        if self.use_omero_state.get():
            obj=self.omero_dataset.option.get()+":"+self.omero_dataset.id.get()
            if REMOTE:
                # TODO: below, still OS dependant 
                _, stdout, stderr = REMOTE.exec_command("cd {}; python -m biom3d.omero_pred --obj {} --bui_dir {} --username {} --password {} --hostname {}".format(
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
                target = self.output_dir.data_dir.get()
                if not os.path.isdir(target):
                    os.makedirs(target, exist_ok=True)
                print("Downloading Omero dataset into", target)
                omero_pred.run(
                    obj=obj,
                    target=target,
                    bui_dir=self.model_selection.logs_dir.get(), 
                    dir_out=self.output_dir.data_dir.get(),
                    user=self.omero_connection.username.get(),
                    pwd=self.omero_connection.password.get(),
                    host=self.omero_connection.hostname.get()
                )
        else: # if not use Omero
            if REMOTE:
                _, stdout, stderr = REMOTE.exec_command("cd {}; python -m biom3d.pred --bui_dir {} --dir_in {} --dir_out {}".format(
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
                pred(
                    bui_dir=self.model_selection.logs_dir.get(),
                    dir_in=self.input_dir.data_dir.get(),
                    dir_out=self.output_dir.data_dir.get())

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

        self.use_proxy.grid(column=0, columnspan=2, row=4, sticky=(W))

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
        window_width = 600
        window_height = 600

        ## get the screen dimension
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

        self.title_label = ttk.Label(self.local_or_remote, text="Biom3d", font=("Montserrat", 18))
        self.welcome_message = ttk.Label(self.local_or_remote, text="Welcome!\n\nBiom3d is an easy-to-use tool to train and use deep learning models for segmenting three dimensional images. You can either start locally, if your computer has a good graphic card (NVIDIA Geforce RTX 1080 or higher) or connect remotelly on a computer with such a graphic card.\n\nIf you need help, check our GitHub repository here: https://github.com/GuillaumeMougeot/biom3d", anchor="w", justify=LEFT, wraplength=450)

        self.start_locally = ttk.Button(self.local_or_remote, text="Start locally", command=lambda: self.main(remote=False))

        self.start_remotelly_frame = Connect2Remote(self.local_or_remote, text="Connect to remote server", padding=[10,10,10,10])
        self.start_remotelly_button = ttk.Button(self.local_or_remote, text='Start remotelly', command=lambda: self.main(remote=True))

        self.title_label.grid(column=0, row=0, sticky=W)
        self.welcome_message.grid(column=0, row=1, sticky=(W,E), pady=12)
        
        # The local button is displayed only for the local installation 
        if LOCAL: 
            self.start_locally.grid(column=0, row=2, sticky=(W,E), pady=12)

        self.start_remotelly_frame.grid(column=0, row=3, sticky=(W,E), pady=12)
        self.start_remotelly_button.grid(column=0, row=4, sticky=(W,E), pady=5)

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

        # Stage 1.2 (root -> root_frame)
        self.root_frame = ttk.Frame(self, padding=PADDING, style='red.TFrame')
        self.root_frame.grid(column=0, row=0, sticky=(N, W, E, S))
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)

        # Stage 2.1  (root_frame -> notebook)
        self.tab_parent = ttk.Notebook(self.root_frame, style='red.TNotebook')
        self.tab_parent.pack(expand=YES, fill='both', padx=6, pady=6)

        # Stage 3 (notebook -> preprocess - train - predict - omero)
        
        self.preprocess_tab = ttk.Frame(self.tab_parent, padding=PADDING)
        self.train_tab = ttk.Frame(self.tab_parent, padding=PADDING)
        self.predict_tab = ttk.Frame(self.tab_parent, padding=PADDING)
        # self.omero_tab = ttk.Frame(self.tab_parent, padding=PADDING)

        self.tab_parent.add(self.preprocess_tab, text="Preprocess")
        self.tab_parent.add(self.train_tab, text="Train")
        self.tab_parent.add(self.predict_tab, text="Predict")

        # Stage 4 (preprocess_tab -> preprocess_tab_frame)
        self.preprocess_tab_frame = PreprocessTab(self.preprocess_tab)
        self.preprocess_tab_frame.grid(column=0, row=0, sticky=(N,W,E), pady=24, padx=12)
        self.preprocess_tab.columnconfigure(0, weight=1)
        self.preprocess_tab.rowconfigure(0, weight=1)

        # Stage 4 (predict_tab -> predict_tab_frame)
        self.predict_tab_frame = PredictTab(self.predict_tab)
        self.predict_tab_frame.grid(column=0, row=0, sticky=(N,W,E), pady=24, padx=12)
        self.predict_tab.columnconfigure(0, weight=1)
        self.predict_tab.rowconfigure(0, weight=1)

        # Stage 4 (train_tab -> train_tab_frame)
        self.train_tab_frame = TrainTab(master=self.train_tab, preprocess_tab=self.preprocess_tab_frame)
        self.train_tab_frame.grid(column=0, row=0, sticky=(N,W,E), pady=24, padx=12)
        self.train_tab.columnconfigure(0, weight=1)
        self.train_tab.rowconfigure(0, weight=1)

if __name__=='__main__':
    parser = argparse.ArgumentParser(description="Graphical User Interface of Biom3d")
    parser.add_argument("-L", "--local", default=True,  action='store_true', dest='local',
        help="Start the GUI with the local version (the remote version is the default version).") 
    args = parser.parse_args()

    LOCAL = args.local

    root = Root()

    try: # avoid blury UI on Windows
        if platform=='win32':
            from ctypes import windll
            windll.shcore.SetProcessDpiAwareness(1)
    finally:
        root.mainloop()

    
