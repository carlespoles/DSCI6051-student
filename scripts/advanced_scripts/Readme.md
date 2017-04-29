## 1 - Setup Datalab.

Install Datalab https://cloud.google.com/datalab/docs/quickstarts and use zone=us-west1-a and project-id ‘wellio-kadaif’ options		  Install Datalab https://cloud.google.com/datalab/docs/quickstarts and use zone=us-west1-a and project-id ‘wellio-kadaif’ options.

## 2 - Running GPU Instances.

To run an instance with a GPU installed follow these instructions.

Update your `gcloud` CLI:
```
gcloud components update && gcloud components install beta
```

Create a new instance with an attached GPU, make sure to be in this projects
root directory.

```
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
   --metadata-from-file startup-script=startup-scripts/install-gpu.sh
```

To start an instance with Jupyter running in Docker:

```
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
   --metadata-from-file startup-script=startup-scripts/install-docker.sh
```

To connect to the running Jupyter notebook use:

```
gcloud compute ssh --zone=us-east1-d \
 --ssh-flag="-D" --ssh-flag="10000" --ssh-flag="-N" --ssh-flag="-n" "gpu-sspyder" &
```

Start Chrome to connect:

```
/Applications/Google\ Chrome.app/Contents/MacOS/Google\ Chrome \
   "http://gpu-sspyder:8888" \
   --proxy-server="socks5://localhost:10000" \
   --host-resolver-rules="MAP * 0.0.0.0 , EXCLUDE localhost" \
   --user-data-dir=/tmp/
```

The password will be **getwellio**.

To delete this instance use:

```
gcloud compute instances delete gpu-sspyder --zone=us-east1-d
```

## 3 - Remember to `chmod 755` the `.sh` scripts.
