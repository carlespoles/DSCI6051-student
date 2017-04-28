#! /bin/bash

# Installing CUDA drivers.
curl -O http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/cuda-repo-ubuntu1604_8.0.61-1_amd64.deb

sudo dpkg -i cuda-repo-ubuntu1604_8.0.61-1_amd64.deb

sudo apt-get update

sudo apt-get install cuda -y

rm cuda-repo-ubuntu1604_8.0.61-1_amd64.deb

nvidia-smi

# You should see on the terminal something like this:
# +-----------------------------------------------------------------------------+
# | NVIDIA-SMI 375.51                 Driver Version: 375.51                    |
# |-------------------------------+----------------------+----------------------+
# | GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
# | Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
# |===============================+======================+======================|
# |   0  Tesla K80           On   | 0000:00:04.0     Off |                    0 |
# | N/A   55C    P0    69W / 149W |      0MiB / 11439MiB |      0%      Default |
# +-------------------------------+----------------------+----------------------+
# |   1  Tesla K80           On   | 0000:00:05.0     Off |                    0 |
# | N/A   54C    P0    70W / 149W |      0MiB / 11439MiB |      0%      Default |
# +-------------------------------+----------------------+----------------------+
# |   2  Tesla K80           On   | 0000:00:06.0     Off |                    0 |
# | N/A   38C    P0    60W / 149W |      0MiB / 11439MiB |      0%      Default |
# +-------------------------------+----------------------+----------------------+
# |   3  Tesla K80           On   | 0000:00:07.0     Off |                    0 |
# | N/A   57C    P0    72W / 149W |      0MiB / 11439MiB |     70%      Default |
# +-------------------------------+----------------------+----------------------+
#
# +-----------------------------------------------------------------------------+
# | Processes:                                                       GPU Memory |
# |  GPU       PID  Type  Process name                               Usage      |
# |=============================================================================|
# |  No running processes found                                                 |
# +-----------------------------------------------------------------------------+

echo 'export CUDA_HOME=/usr/local/cuda' >> ~/.bashrc

echo 'export PATH=$PATH:$CUDA_HOME/bin' >> ~/.bashrc

echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CUDA_HOME/lib64' >> ~/.bashrc

source ~/.bashrc

# Installing CUDA NN library (it was downloaded from NVIDIA. Now, in a bucket).
gsutil cp gs://wellio-kadaif-cuda/cudnn-8.0-linux-x64-v5.1.tgz .

tar xzvf cudnn-8.0-linux-x64-v5.1.tgz

sudo cp cuda/lib64/* /usr/local/cuda/lib64/

sudo cp cuda/include/cudnn.h /usr/local/cuda/include/

sudo apt-get update

# Installing python, etc.

sudo apt-get -y install python2.7 python-pip python-dev

sudo apt-get -y install ipython ipython-notebook

sudo -H pip install --upgrade pip

sudo -H pip install jupyter

sudo pip install tensorflow-gpu

sudo pip install keras

sudo pip install pandas h5py sklearn

sudo apt-get install python-matplotlib -y

sudo pip install --upgrade google-cloud-storage

jupyter notebook --generate-config

# Note the location of an output like:
# Writing default config to: /home/carles/.jupyter/jupyter_notebook_config.py
# As we need to edit that file.

echo "Finished installing tensorflow-gpu, keras, pandas, h5py, sklearn, ipython, jupyter and matplotlib."
