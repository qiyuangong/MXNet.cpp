#!/bin/bash
rm -f hand_written
make hand_written
export PS_VERBOSE=1
../../mxnet/tools/launch.py -n 2 --launcher local ./hand_written dist
