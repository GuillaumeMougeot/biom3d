#!/bin/sh
submodule=$1
shift
if [ "$TESTED" = 0 ]; then
    echo "[WARNING] This image hasn't been totally tested. If any problem is encountered, please open an issue at https://github.com/GuillaumeMougeot/biom3d"
fi

exec python3.11 -m biom3d."$submodule" "$@"
