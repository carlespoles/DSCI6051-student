gcloud beta compute instances create \
gpu-sspyder --machine-type n1-standard-2 \
--zone us-east1-d \
--accelerator type=nvidia-tesla-k80,count=1 \
--image-family ubuntu-1604-lts \
--image-project ubuntu-os-cloud \
--scopes bigquery,datastore,logging-write,storage-full,https://www.googleapis.com/auth/pubsub \
--boot-disk-size 200GB \
--maintenance-policy TERMINATE \
--restart-on-failure \
--metadata-from-file startup-script=install-docker.sh
