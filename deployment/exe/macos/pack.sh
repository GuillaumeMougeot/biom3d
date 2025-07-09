#!/bin/bash

set -e

ENV_NAME=installer
DIR="Biom3d"
ARCH="arm64"
if [ ! -z "$1" ]; then
    ARCH="$1"
fi
DIR="${DIR}_${ARCH}.app"

# Check if environment exists
if conda env list | grep -i "^${ENV_NAME}[[:space:]]" > /dev/null; then
    echo "Environment '${ENV_NAME}' already exists."
else
    echo "Create environment '${ENV_NAME}'..."
    "$CONDA_PATH" create -y -n "$ENV_NAME" python=3.11 tk
fi

# Activate environment
source "$(conda info --base)/etc/profile.d/conda.sh"
source conda activate "$ENV_NAME"

# Avoid pip/conda conflicts
conda install -y pip=23.1

# Omero dependencies
conda install -y zeroc-ice=3.6.5

pip install pillow future portalocker requests "urllib3<2"
# pywin32 n'est pas disponible ni utile sous macOS

# pip install omero-py sans dépendances car zeroc-ice est déjà installé
pip install --no-deps omero-py
pip install --no-deps ezomero
pip install biom3d
pip cache purge

# Pack
if [ -d "$DIR" ]; then
    echo "Folder $DIR already exists, deleting..."
    rm -rf "$DIR"
fi
mkdir -p "$DIR"
mkdir -p "$DIR/Contents"
DIR="$DIR/Contents"
mkdir -p "$DIR/MacOS"
mkdir -p "$DIR/Resources"
echo "FIRST_LAUNCH=1" > "$DIR/MacOS/bin/.env"

# conda pack (change output path)
conda pack --format=no-archive -o "$DIR/bin"

conda deactivate

# Copier fichiers
cp Biom3d.sh "$DIR/MacOS/Biom3d.sh"
cp logo.icns "$DIR/Resources/Biom3d.icns"
cp Info.plist "$DIR/Info.plist"
chmod +x "$DIR/MacOS/Biom3d.sh"

# Zip (7z if installed, else default zip)
if command -v 7z >/dev/null 2>&1; then
    7z a -tzip Biom3d.zip "$DIR"
else
    # zip natif macOS
    zip -r Biom3d.zip "$DIR"
fi
