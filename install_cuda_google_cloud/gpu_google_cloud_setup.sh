#! /bin/bash

declare -r MACHINE_NAME="gpu-images"
declare -r MACHINE_TYPE="n1-highmem-8"
declare -r ZONE="us-east1-d"
declare -r GPU_COUNT="4"
declare -r OS="ubuntu-1604-lts"
declare -r DISK_SIZE="200GB"

echo "Creating GPU instance with name: " $MACHINE_NAME
echo "Machine type: " $MACHINE_TYPE
echo "Creating on zone: " $ZONE
echo "Number of GPUs: " $GPU_COUNT
echo "Machine OS: " $OS
echo "Machine disk size: " $DISK_SIZE

# gcloud compute instances delete $MACHINE_NAME

gcloud beta compute instances create $MACHINE_NAME \
  --machine-type $MACHINE_TYPE \
  --zone $ZONE \
  --accelerator type=nvidia-tesla-k80,count=$GPU_COUNT \
  --image-family $OS \
  --image-project ubuntu-os-cloud \
  --boot-disk-size $DISK_SIZE \
  --maintenance-policy TERMINATE \
  --restart-on-failure

# Wait for confirmation of instance creation, something like:
# NAME        ZONE        MACHINE_TYPE  PREEMPTIBLE  INTERNAL_IP  EXTERNAL_IP     STATUS
# gpu-images  us-east1-d  n1-highmem-8               10.142.0.2   35.185.122.94
