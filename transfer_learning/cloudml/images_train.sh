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

# Usage: we could pass BUCKET_NAME as first argument and GCS_PATH as second argument.
# Without having to pass the above arguments, we run from the command line from the
# directory this script lives:
# ./images_train.sh
# If arguments were passed, then:
# ./images_train.sh BUCKET_NAME GCS_PATH

# if [ -z "$1" ]
#   then
#     echo "No bucket supplied."
#     exit 1
# fi

# if [ -z "$2" ]
#   then
#     echo "No GCS path supplied"
#     exit 1
# fi

#declare -r BUCKET=$1
#declare -r GCS_PATH=$2

if [ -z "$3" ]
  then
    declare -r JOB_NAME="food_${USER}_$(date +%Y%m%d_%H%M%S)"
  else
    declare -r JOB_NAME=$3
fi

BUCKET_NAME='gs://wellio-kadaif-tasty-images-project-images'
OUTPUT_PATH=$BUCKET_NAME/$JOB_NAME

echo
echo "Using bucket: " $BUCKET
echo "Using job id: " $JOB_NAME
echo "Using output path: " $OUTPUT_PATH

set -v -e

# Train on cloud ML. GPU's are only available in region us-east1
gcloud ml-engine jobs submit training "$JOB_NAME" \
  --package-path trainer/ \
  --module-name trainer.task \
  --runtime-version=1.0 \
  --job-dir $OUTPUT_PATH \
  --region us-east1 \
  --config=trainer/config.yaml \
  --output_path "${OUTPUT_PATH}/training" \
  --eval_data_paths "${OUTPUT_PATH}/preproc/eval*" \
  --train_data_paths "${OUTPUT_PATH}/preproc/train*"
  -- \

# You can also separately run:
# gcloud beta ml jobs stream-logs "$JOB_ID"
# to see logs for a given job.

set +v
