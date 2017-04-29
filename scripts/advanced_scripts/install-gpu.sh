#! /bin/bash
 
 # Installs Tensorflow with GPU support
 sudo su -
 
 curl -O http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/cuda-repo-ubuntu1604_8.0.61-1_amd64.deb
 dpkg -i cuda-repo-ubuntu1604_8.0.61-1_amd64.deb
 apt-get update
 rm cuda-repo-ubuntu1604_8.0.61-1_amd64.deb
 apt-get install cuda -y
 
 # confirm Nvidia support
 nvidia-smi
 
 echo 'export CUDA_HOME=/usr/local/cuda' >> ~/.bashrc
 echo 'export PATH=$PATH:$CUDA_HOME/bin' >> ~/.bashrc
 echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CUDA_HOME/lib64' >> ~/.bashrc
 
 # download Cuda NN library
 gsutil cp gs://kadaif.cephalo.ai/cuda/cudnn-8.0-linux-x64-v5.1.tgz .
 tar xzvf cudnn-8.0-linux-x64-v5.1.tgz
 sudo cp cuda/lib64/* /usr/local/cuda/lib64/
 
 ## Python Setup
 
 # curl -O https://repo.continuum.io/archive/Anaconda3-4.3.1-Linux-x86_64.sh
 # bash Anaconda3-4.3.1-Linux-x86_64.sh -b
 
 apt-get update
 apt-get -y install python2.7 python-pip python-dev
 export TF_BINARY_URL=https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow-0.11.0-cp27-none-linux_x86_64.whl
 pip install --upgrade $TF_BINARY_URL
 pip install keras