# Instructions on submitting a job using ml-engine.

## 1 - `trainer` folder.

It needs to contain a `.yaml` file where we specify the GPUs to use. Note that as of the time of this writting, they are only available in region `us-east1`.

Besides the `.yaml` file, the `task.py` is the script where our model is described, dataset is loaded from a bucket, and it will be trained and evaluated.

## 2 - `setup.py` file.

Note that the file `setup.py` is one level up the `trainer folder`, and its relevance its due to the fact that we specify the packages required t run the `task.py` script:

```
REQUIRED_PACKAGES = [
  'tensorflow==1.0.1',
  'keras==2.0.3',
]
```
The job is submitted using the command line as it follows (make sure we are in the `trainer` directory):

```
gcloud ml-engine jobs submit training $JOB_NAME \
--job-dir $OUTPUT_PATH \
--runtime-version 1.0 \
--module-name trainer.task \
--package-path trainer/ \
--region $REGION \
--config=trainer/config.yaml
-- \
```

`--module-name trainer.task` specifies the script we need to run: it's the `task` file under the `trainer` folder. If we had a different script to run under the `trainer` folder instead of `task,py`, for example `another_script.py`, then we would change that part of the `ml-engine` job as:

`--module-name trainer.another_script`

Note that in our case, we will specify:

`OUTPUT_PATH = gs://wellio-kadaif-tasty-images-ml-engine/jobs`

(the folder `jobs` needs to already exist in the bucket `gs://wellio-kadaif-tasty-images-ml-engine`)

`JOB_NAME` needs to be unique each time run this job, so it needs to be changed on the command line before submitting it.

`JOB_NAME = job_image_classification_1`

`REGION = us-east1` since GPUs only run on that region.

Effectively, here is what submit in the command line:

```
gcloud ml-engine jobs submit training 'job_image_classification_1' \
--job-dir 'gs://wellio-kadaif-tasty-images-ml-engine/jobs' \
--runtime-version 1.0 \
--module-name trainer.task \
--package-path trainer/ \
--region 'us-east1' \
--config=trainer/config.yaml
```

For example:

```
(wellio) Admins-MacBook-Pro:DSCI6051-student carles$ cd ml_engine/
(wellio) Admins-MacBook-Pro:ml_engine carles$ ls
Readme.md	images		setup.py	trainer
(wellio) Admins-MacBook-Pro:ml_engine carles$ cd cd trainer/
-bash: cd: cd: No such file or directory
(wellio) Admins-MacBook-Pro:ml_engine carles$ gcloud ml-engine jobs submit training 'job_image_classification_1' \
> --job-dir 'gs://wellio-kadaif-tasty-images-ml-engine/jobs' \
> --runtime-version 1.0 \
> --module-name trainer.task \
> --package-path trainer/ \
> --region 'us-east1' \
> --config=trainer/config.yaml
Job [job_image_classification_1] submitted successfully.
Your job is still active. You may view the status of your job with the command

  $ gcloud ml-engine jobs describe job_image_classification_1

or continue streaming the logs with the command

  $ gcloud ml-engine jobs stream-logs job_image_classification_1
jobId: job_image_classification_1
state: QUEUED
(wellio) Admins-MacBook-Pro:ml_engine carles$
```

## 3 - Run job as bash command.

As an alternative, we can run `submit_ml.sh` from the command line:

`./submit_ml.sh`

Here is an example:

```
(wellio) Admins-MacBook-Pro:ml_engine carles$ ./submit_ml.sh

Using JOB_NAME:  job_image_classification_carles_20170425_085655
Using OUTPUT_PATH:  gs://wellio-kadaif-tasty-images-ml-engine/jobs
Using REGION:  us-east1

gcloud ml-engine jobs submit training $JOB_NAME \
--job-dir $OUTPUT_PATH \
--runtime-version 1.0 \
--module-name trainer.task \
--package-path trainer/ \
--region $REGION \
--config=trainer/config.yaml
Job [job_image_classification_carles_20170425_085655] submitted successfully.
Your job is still active. You may view the status of your job with the command

  $ gcloud ml-engine jobs describe job_image_classification_carles_20170425_085655

or continue streaming the logs with the command

  $ gcloud ml-engine jobs stream-logs job_image_classification_carles_20170425_085655
jobId: job_image_classification_carles_20170425_085655
state: QUEUED
```

Note that the file `submit_ml.sh` needs to have the right permissions: `chmod 755 submit_ml.sh`.

The job can be monitored using Google Cloud ML Engine GUI:

<img src='images/ml_engine_0.jpg' />

then, click on `view logs`:

<img src='images/ml_engine.jpg' />
