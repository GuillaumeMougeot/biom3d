#!/bin/bash

# Quit immediately on error
set -e

# Load environment variables from env.sh
source ./bin/env.sh

# Check if it is the first launch
if [ "$FIRST_LAUNCH" = "1" ]; then
    echo "First launch detected, initializing the virtual environment"

    # Run conda-unpack if it's available
    if [ -x "./bin/bin/conda-unpack" ]; then
        ./bin/bin/conda-unpack
    else
        echo "conda-unpack not found!" >&2
        exit 1
    fi

    echo "Virtual environment initialized"
    echo "export FIRST_LAUNCH=0" > ./bin/env.sh   

echo "Starting Biom3d..."
# Launch Biom3d GUI
./bin/bin/python3.11 -c ./bin/bin/python3.11 -m biom3d.gui
