
# create a venv :
python -m venv NAME_OF_YOUR_VIRTUAL_ENV
# then install :
pip install pyinstaller python-tk numpy SimpleITK scikit-image tqdm paramiko pyyaml matplotlib 
# run the pyinstaller (with console) and exclude modules
pyinstaller -F --clean --upx-dir PATH_TO_UPX   PATH_TO_GUI\gui.py --exclude=biom3d --exclude=torch --hidden-import='PIL._tkinter_finder'