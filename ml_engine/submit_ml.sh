#!/bin/bash

declare -r JOB_NAME="job_image_classification_${USER}_$(date +%Y%m%d_%H%M%S)"
declare -r BUCKET="gs://wellio-kadaif-tasty-images-ml-engine"
declare -r JOB_FOLDER="jobs"
declare -r OUTPUT_PATH="${BUCKET}/${JOB_FOLDER}"
declare -r REGION="us-east1"

echo
echo "Using JOB_NAME: " $JOB_NAME
echo "Using OUTPUT_PATH: " $OUTPUT_PATH
echo "Using REGION: " $REGION
set -v -e

gcloud ml-engine jobs submit training $JOB_NAME \
--job-dir $OUTPUT_PATH \
--runtime-version 1.0 \
--module-name trainer.task \
--package-path trainer/ \
--region $REGION \
--config=trainer/config.yaml
