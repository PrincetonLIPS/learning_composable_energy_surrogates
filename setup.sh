#!/bin/bash
source ~/.bashrc
source ~/anaconda2/etc/profile.d/conda.sh
source ~/anaconda3/etc/profile.d/conda.sh
conda remove --name nm --all
conda create -n nm python==3.6.3
conda activate nm
conda install -c conda-forge dolfin-adjoint==2019.1.0
conda install -c conda-forge mshr
conda install pytorch torchvision -c pytorch
conda install -c conda-forge hdf5==1.10.4
conda install -c conda-forge numpy scipy matplotlib jupyter # pip install numpy scipy matplotlib jupyter
pip install --upgrade ray==0.7.4
pip install --upgrade tensorflow
pip install boto3
