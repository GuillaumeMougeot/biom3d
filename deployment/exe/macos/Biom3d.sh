#!/bin/bash

# Quit immediately on error
set -e

# Load environment variables from env.sh
source ./bin/env.sh

# Check if it is the first launch
if [ "$FIRST_LAUNCH" = "1" ]; then
    echo "First launch detected, initializing the virtual environment"

    # Run conda-unpack if it's available
    if [ -x "./bin/Scripts/conda-unpack" ]; then
        ./bin/Scripts/conda-unpack
    elif [ -x "./bin/conda-unpack" ]; then
        ./bin/conda-unpack
    else
        echo "conda-unpack not found!" >&2
        exit 1
    fi

    echo "Virtual environment initialized"
    # Overwrite env.sh to set FIRST_LAUNCH=0
    cat > ./bin/env.sh <<EOF
#!/bin/bash
export FIRST_LAUNCH=0
EOF
    chmod +x ./bin/env.sh
fi

echo "Starting Biom3d..."
# Launch Biom3d GUI
./bin/python -m biom3d.gui
