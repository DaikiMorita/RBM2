#!/bin/bash

python src/ConfigGenerator.py Config/ConfigOriginal/ConfigOriginal.ini

if [ $? = 0 ]; then
    bash RBM_RUN.sh
else
    echo "Finished"
fi
