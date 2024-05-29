#!/bin/bash

if ! command -v conda &> /dev/null
then
    echo "Conda could not be found. Install Conda to proceed."
    exit
fi

conda info --envs &> /dev/null
if [ $? -ne 0 ]; then
    echo "The shell is not properly configured to use 'conda activate'."
    echo "Initializing shell for Conda..."
    conda init $(basename $SHELL)
    echo "Please close and restart terminal, then run this script again."
    exit
fi

if [ -d "SMICNet_env" ]; then
    rm -rf SMICNet_env
fi

conda create --name SMICNet_env python=3.8 -y


source $(conda info --base)/etc/profile.d/conda.sh
conda activate SMICNet_env


if [ $? -ne 0 ]; then
    echo "Failed to activate Conda environment."
    exit
fi


conda install numpy==1.21.2 -y
if [ $? -ne 0 ]; then
    echo "Failed to install numpy using Conda."
    exit
fi

pip install --upgrade pip
if [ $? -ne 0 ]; then
    echo "Failed to upgrade pip."
    exit
fi

pip install -r requirements_mac.txt
if [ $? -ne 0 ]; then
    echo "Failed to install packages from requirements_mac.txt."
    exit
fi

python -m pip list | grep numpy
python -m pip list | grep -f requirements_mac.txt

python -m ipykernel install --user --name=SMICNet_env --display-name="SMICNet_env"

echo "Environment setup is complete."
