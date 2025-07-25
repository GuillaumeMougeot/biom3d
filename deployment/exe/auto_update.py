import subprocess
import re
import sys

def get_cuda_version_from_nvcc():
    output = subprocess.check_output(["nvcc", "--version"], stderr=subprocess.STDOUT, text=True)
    # Exemple de sortie : "Cuda compilation tools, release 11.8, V11.8.89"
    match = re.search(r"release (\d+)\.(\d+)", output)
    if match:
        major = match.group(1)
        return int(major)

def get_cuda_version_from_nvidia_smi():
    try:
        output = subprocess.check_output(["nvidia-smi"], stderr=subprocess.STDOUT, text=True)
        # Cherche une ligne comme : "CUDA Version: 12.2"
        match = re.search(r"CUDA Version: (\d+)\.(\d+)", output)
        if match:
            major = match.group(1)
            return int(major)
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None

def detect_cuda_major_version():
    try :
        version = get_cuda_version_from_nvcc()
    except :
        try :
            version = get_cuda_version_from_nvidia_smi()
        except :
            version = None

    if version is not None :
        # Install 11.8 or 12.8 (we use the x.8 retrocompatibility)
        subprocess.check_call([sys.executable, "-m", "pip", "install", "torch","--index-url","https://download.pytorch.org/whl/cu"+str(version)+"8","--force-reinstall","--no-warn-script-location","--no-deps"]) #Remove no-deps once typing-extnsions doesn't bug

if __name__ == "__main__":
    detect_cuda_major_version()