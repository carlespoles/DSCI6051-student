#!/bin/bash
# Copyright 2016 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# This sample assumes you're already setup for using CloudML.  If this is your
# first time with the service, start here:
# https://cloud.google.com/ml/docs/how-tos/getting-set-up

# Now that we are set up, we can start processing some food images.
declare -r PROJECT=$(gcloud config list project --format "value(core.project)")
declare -r JOB_ID="tasty_images_${USER}_$(date +%Y%m%d_%H%M%S)"
declare -r BUCKET="gs://wellio-kadaif-tasty-images-project-images"
declare -r GCS_PATH="${BUCKET}/${USER}/${JOB_ID}"
declare -r DICT_FILE=gs://wellio-kadaif-tasty-images-project-images/dict.txt
declare -r MODEL_NAME=tasty_images
declare -r VERSION_NAME=v1

echo
echo "Using job id: " $JOB_ID
echo "Using GCS_PATH: " $GCS_PATH
set -v -e

# Takes about 30 mins to preprocess everything.  We serialize the two
# preprocess.py synchronous calls just for shell scripting ease; you could use
# --runner DataflowRunner to run them asynchronously.  Typically,
# the total worker time is higher when running on Cloud instead of your local
# machine due to increased network traffic and the use of more cost efficient
# CPU's.  Check progress here: https://console.cloud.google.com/dataflow
python trainer/preprocess.py \
  --input_dict "$DICT_FILE" \
  --input_path "gs://wellio-kadaif-tasty-images-project-images/eval_images_path_set.csv" \
  --output_path "${GCS_PATH}/preproc/eval" \
  --num_workers 100 \
  --cloud

python trainer/preprocess.py \
  --input_dict "$DICT_FILE" \
  --input_path "gs://wellio-kadaif-tasty-images-project-images/train_images_path_set.csv" \
  --output_path "${GCS_PATH}/preproc/train" \
  --num_workers 100 \
  --cloud

# Training on CloudML is quick after preprocessing. If you ran the above
# commands asynchronously, make sure they have completed before calling this one.
gcloud ml-engine jobs submit training "$JOB_ID" \
  --stream-logs \
  --module-name trainer.task \
  --package-path trainer \
  --staging-bucket "$BUCKET" \
  --region us-east1 \
  --config=trainer/config.yaml \
  --runtime-version=1.0 \
  -- \
  --output_path "${GCS_PATH}/training" \
  --eval_data_paths "${GCS_PATH}/preproc/eval*" \
  --train_data_paths "${GCS_PATH}/preproc/train*"

# Tell CloudML about a new type of model coming.  Think of a "model" here as
# a namespace for deployed Tensorflow graphs.
gcloud ml-engine models create "$MODEL_NAME" \
  --regions us-east1

# Each unique Tensorflow graph--with all the information it needs to execute--
# corresponds to a "version".  Creating a version actually deploys our
# Tensorflow graph to a Cloud instance, and gets is ready to serve (predict).
gcloud ml-engine versions create "$VERSION_NAME" \
  --model "$MODEL_NAME" \
  --origin "${GCS_PATH}/training/model" \
  --runtime-version=1.0

# Models do not need a default version, but its a great way move your production
# service from one version to another with a single gcloud command.
gcloud ml-engine versions set-default "$VERSION_NAME" --model "$MODEL_NAME"

# Finally, download a sample image and so we can test online prediction.
gsutil cp \
  gs://wellio-kadaif-tasty-images-project-images/predict_images/trump_1.jpg \
  donald_trump.jpg

# Since the image is passed via JSON, we have to encode the JPEG string first.
python -c 'import base64, sys, json; img = base64.b64encode(open(sys.argv[1], "rb").read()); print json.dumps({"key":"0", "image_bytes": {"b64": img}})' donald_trump.jpg &> request.json

# Here we are showing off CloudML online prediction which is still in beta.
# If the first call returns an error please give it another try; likely the
# first worker is still spinning up.  After deploying our model we give the
# service a moment to catch up--only needed when you deploy a new version.
# We wait for 10 minutes here, but often see the service start up sooner.
sleep 10m
gcloud ml-engine predict --model ${MODEL_NAME} --json-instances request.json
