#! /bin/bash
# Installs Docker with GPU support
sudo su -

cd /root

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
gsutil cp gs://kadaif.cephalo.ai/cuda/cudnn-8.0-linux-x64-v6.0.tgz .
tar xzvf cudnn-8.0-linux-x64-v6.0.tgz
cp cuda/lib64/* /usr/local/cuda/lib64/

## Docker setup
apt-get install -y \
      linux-image-extra-$(uname -r) \
       linux-image-extra-virtual

apt-get install -y \
        apt-transport-https \
        ca-certificates \
        curl \
        software-properties-common

curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -

 add-apt-repository \
    "deb [arch=amd64] https://download.docker.com/linux/ubuntu \
   $(lsb_release -cs) \
   stable"

 apt-get update
 apt-get install -y docker-ce

 # Install nvidia-docker and nvidia-docker-plugin
 wget -P /tmp https://github.com/NVIDIA/nvidia-docker/releases/download/v1.0.1/nvidia-docker_1.0.1-1_amd64.deb
 sudo dpkg -i /tmp/nvidia-docker*.deb && rm /tmp/nvidia-docker*.deb

 # Test nvidia-smi
 nvidia-docker run --rm nvidia/cuda nvidia-smi

 # Start Jupyter notebook
 nvidia-docker run -d -p 8888:8888 -e PASSWORD=getwellio gcr.io/tensorflow/tensorflow:latest-gpu
