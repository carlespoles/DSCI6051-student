
# Classifying images using *transfer learning*.

  - [Setup in Google Storage](#set-up-in-google-storage).
  - [Execution of the script](#execution-of-the-script).
    - [1. Image Preprocessing](##1-image-preprocessing-details).
    - [2. Modeling: Training the model](##2-training-and-monitoring-the-model).

  - [The "tasty/not-tasty" image classification task](###the "tasty/not-tasty" image classification task)
    - [1. Image Preprocessing](#1-image-preprocessing)
      - [1.1 Deploy the preprocessing job to Cloud Dataflow](#11-deploy-the-preprocessing-job-to-cloud-dataflow)
    - [2. Modeling: Training the classifier](#2-modeling-training-the-classifier)
      - [2.1 For the workshop, use pre-generated TFRecords for training](#21-for-the-workshop-use-pre-generated-tfrecords-for-training)
      - [2.2 Run the training script](#22-run-the-training-script)
      - [2.3 Monitor the training](#23-monitor-the-training)
    - [3. Prediction: Using the trained model](#3-prediction-using-the-trained-model)
      - [3.1 Prediction from the command line using gcloud](#31-prediction-from-the-command-line-using-gcloud)
      - [3.2 Prediction using the Cloud ML API: A prediction web server](#32-prediction-using-the-cloud-ml-api-a-prediction-web-server)
  - [Appendix: Running training locally](#appendix-running-training-locally)

Using [Google Vision API](https://cloud.google.com/vision/) is a good resource to identify labels, or categories, for a given image. The problem can arise when we need to further classify your own images, in more specialized categories that the Google Vision API hasn't been trained on.

This project shows how an existing neural network can be used to accomplish the above task using *transfer learning* which bootstraps an existing model to reduce the effort needed to learn something new.

The 'Inception v3' architecture model trained to classify images against 1,000 different 'ImageNet' categories, and using its penultimate "bottleneck" layer, is used to train train a new top layer that can recognize other classes of images, like "tasty" or "not-tasty" in this project.

The new top layer does not need to be very complex, and that we typically don't need much data or much
training of this new model, to get good results for our new image classifications.

![Transfer learning](images/image-classification-3-1.png)

Besides transfer learning, we show how to use other aspects of TensorFlow and Cloud ML. It shows how to use
[Cloud Dataflow](https://cloud.google.com/dataflow/) ([Apache Beam](https://beam.apache.org/))
to do image preprocessing -- the Beam pipeline uses Inception v3 to generate the inputs to the new 'top layer' that we will train and will use the preprocessing results (TFRecords) to be consumed during training.

This project also includes a tiny "prediction web server" using Flask that uses **Cloud ML API for prediction** once the trained model is serving.

To run the process end to end, we need to run this shell script on the command line: `./tasty_images.sh`

As a **reminder**, we need to first create an anaconda environment with python 2.7.

Then, after creating the environment, we need to setup Google Cloud Platform SDK:

Install Google Cloud Platform SDK https://cloud.google.com/sdk/downloads and follow the instructions for `interactive installer`.

We will also need to install Tensorflow.

Below, all steps executed in the script are described step by step.

## Set up in Google Storage.

The images we downloaded need to be stored in a bucket:

For this project, the bucket is `gs://wellio-kadaif-tasty-images-project-images`

![For images with the label 'ok' (or 'tasty')](images/bucket-1.jpg)

![For images with the label 'nok' (or 'non-tasty')](images/bucket-2.jpg)

**All required files to pre-process images can be found on the `input_files` folder.**

We need to create a `.csv` file with the path to each image (`all_images_path.csv`), as well as the it's corresponding label. Here is how the file looks like:

![Path for images with the label 'ok' (or 'tasty')](images/path-1.jpg)

![Path for images with the label 'nok' (or 'non-tasty')](images/path-2.jpg)

From the previous `.csv` file we create a training and validation `.csv` files, using a 90/10 percent split and randomly assigning images to each set (training/validation).

The files are called respectively `train_images_path_set.csv`and `eval_images_path_set.csv`.

Finally, we create a `dict.txt` file, which is a dictionary of the labels we require in our project:

![Dictionary labels file](images/labels.jpg)

Finally, we load `train_images_path_set.csv`, `eval_images_path_set.csv` and `dict.txt` to the same bucket:

![All files uploaded](images/bucket-3.jpg)

## Execution of the script.

Ensure that the script permissions are the correct ones to be executed. If an error is thrown, then have them changed by issuing  `chmod 755 tasty_images.sh`.

We need to set up some default values in the above script:

![View of script](images/shell-1.jpg)

The script sends to the screen the `GCS_PATH` for convenience, and we should make not of it as will need it to reference it when launching `tensorboard`.

Note that since hardcode `VERSION_NAME`, if the script is run again to train a new model, it needs to be changed or it will throw an error.

Before start running the script, we need to `source activate MY_ENVIRONMENT` first. In this project `MY_ENVIRONMENT` is `wellio`.

As mentioned at the beginning, we execute `./tasty_images.sh` at the command line.

![Run the shell](images/tut-1.jpg)

Note that we specified `--num-workers` to be `100` to get more nodes to process the job. If not specified, it's defaulted to 10 workers.

As soon as the script starts to run, it will start by pre-processing of the images specified on the evaluation `.csv` set of images using Google Dataflow.

![View of script](images/shell-2.jpg)

Dataflow can be monitored here <https://console.cloud.google.com/dataflow?project=wellio-kadaif>

and by clicking on the highlighted square, we can see the details of the job:

![Monitor dataflow](images/flow-1.jpg)
![Monitor dataflow](images/flow-2.jpg)
![Monitor dataflow](images/flow-3.jpg)

As soon as the evaluation set pre-processing is completed, a new dataflow job starts for the training set.

![Run the shell](images/tut-2.jpg)
![View of script](images/shell-3.jpg)

Once pre-processing is completed, we will have two completed jobs.

![Monitor dataflow](images/flow-4.jpg)

## 1. Image preprocessing details.

Preprocessing means that we extract an image features from the "bottleneck" layer which is the penultimate layer of the Inception network. This is achieved by loading the saved Inception model and its variable values into TensorFlow, and run each image through that model, which has been open-sourced by Google).

Each image is processed to produce its feature representation (an *embedding*) which is a k-dimensional vector of floats (in our case, 2,048 dimensions). The preprocessing includes converting the image format, resizing images, and running the converted image through a pre-trained model to get the embeddings.

The reason this approach is so effective for bootstrapping new image classification is that these 'bottleneck'
embeddings contain a lot of high-level feature information useful to InceptionV3 for its own image classification.

Once the pre-processing has been completed, the embeddings can be found in the bucket (which is under `GCS_PATH/preproc`):

![Embeddings](images/embed.jpg)

## 2. Training and monitoring the model.

Once pre-processing has been completed, the script will continue to run, this time to train the model.

![Run the shell](images/tut-3.jpg)
![View of script](images/shell-4.jpg)

Note that in the folder `trainer`, `task.py` will create the model, and that the `config.yaml` file contains the details of the GPUs to use (only available in Google Cloud `east` region as of the time of this writing).

The `.yaml` file has a link to all possible machines available to run this part of the script.

The neural network will comprise a single fully- connected layer with *RELU* activations and with one output for each label in the dictionary (`dict.txt`) to replace the original output layer.

The final output is computed using the [softmax](https://en.wikipedia.org/wiki/Softmax_function) function. In the
training stages, we use [dropout](https://en.wikipedia.org/wiki/Dropout_(neural_networks)) to randomly ignore a subset of input weights to prevent over-fitting to the training dataset.

The script will output a summary and a model checkpoint information under `$GCS_PATH/training`.

The job can be monitored by looking at the logs here: <https://console.cloud.google.com/mlengine/jobs?project=wellio-kadaif>

![Monitor ML](images/log-0.jpg)
![Monitor ML](images/log-1.jpg)

We also can monitor the accuracy of the training and evaluation sets:

![Monitor ML](images/tut-4.jpg)
![Monitor ML](images/tut-5.jpg)
![Monitor ML](images/tut-6.jpg)

Once the model has been trained, it will be saved in the step of the script.

## 3. Saving the model.

The next step of the script will save the model created previously:

![View of script](images/shell-5.jpg)
![Monitor ML](images/tut-7.jpg)

The model can be found here: <https://console.cloud.google.com/mlengine/models?project=wellio-kadaif>
![Monitor ML](images/log-2.jpg)
![Monitor ML](images/log-3.jpg)

#### 2.3 Monitor the training

As the training runs, you should see the logs stream to STDOUT.  You can also view them with:

```shell
gcloud beta ml jobs stream-logs "$JOB_ID"
```

We can also monitor the progress of the training using [Tensorboard](https://www.tensorflow.org/how_tos/summaries_and_tensorboard/).  

To do this, start up Tensorboard in a new shell (don't forget to activate your virtual environment), pointing it to the training logs in GCS:

```shell
tensorboard --logdir=$GCS_PATH/training
```

Then, visit `http://localhost:6006`.


### 3. Prediction: Using the trained model

For prediction, we don't want to separate the image preprocessing and inference into two separate steps, because we need to perform both in sequence for every image. Instead, we create a single TensorFlow graph that produces the image embedding and does the classification using the trained model in one step.

After training, the saved model information will be in `$GCS_PATH/training/model`.

Our next step is to tell Cloud ML that we want to use and serve that model.
We do that via the following script, where `v1` is our model version name, and `hugs` is our model name.


```shell
./model.sh $GCS_PATH v1 hugs
```

The model is created first.  This only needs to happen once, and is done as follows:
`gcloud beta ml models create <model_name>`

Then, we create a 'version' of that model, based on the data in our model directory (`$GCS_PATH/training/model`), and set that version as the default.

You can see what models and default versions we have in your project via:

```shell
gcloud beta ml models list
```

It will take a minute or so for the model version to start "serving".  Once our model is serving, we make prediction requests to it -- both from the command line and via the Cloud ML API.


#### 3.1 Prediction from the command line using gcloud

To make a prediction request from the command line, we need to encode the image(s) we want to send it into a json format.  See the `images_to_json.py` script for the details.  This command:

```
python images_to_json.py -o request.json <image1> <image2> ...
```

results in a `request.json` file with the encoded image info. Then, run this command:

```shell
gcloud beta ml predict --model $MODEL_NAME --json-instances request.json
```

You should see a result something along the lines of the following:

```shell
gcloud beta ml predict --model hugs --json-instances request.json
KEY                             PREDICTION  SCORES
prediction_images/hedgehog.jpg  1           [4.091006485396065e-05, 0.9999591112136841, 1.8843516969013763e-08]
```

The prediction index (e.g. '1') corresponds to the label at that index in the 'label dict' used to construct the example set during preprocessing,
and the score for each index is listed under SCORES. (The last element in the scores list is used for any example images that did not have an associated label).

The 'hugs' dataset label dict is here:

```shell
gsutil cat gs://oscon-tf-workshop-materials/transfer_learning/cloudml/hugs_photos/dict.txt
```

So, that means that index 0 is the 'hugs' label, and index 1 is 'not-hugs'.  Therefore, the prediction above indicates that the hedgehog is 'not-hugs', with score 0.9999591112136841.

#### 3.2 Prediction using the Cloud ML API: A prediction web server

We can also request predictions via the Cloud ML API, and the google api client libraries.
We've included a little example web server that shows how to do this, and also makes it easy to see how a given image gets labeled.

The web app lets you upload an image, and then it generates a json request containing the image, and sends that to the Cloud ML API.  In response, we get back prediction and score information, and display that on a result page.

Run the web server according to the instructions in its [README](web_server).

## Appendix: Running training locally

If you want to run the training session locally (this can be useful for debugging), you will need to point the
`GOOGLE_APPLICATION_CREDENTIALS` environment variable to a local service account credentials file like this:

```shell
export GOOGLE_APPLICATION_CREDENTIALS=/path/to/credentials/file.json
```

Then initiate the local training like this, defining a local output path to use:

```shell
gcloud beta ml local train --package-path trainer/ --module-name trainer.task \
    -- \
    --max-steps 1000 \
    --train_data_paths "$GCS_PATH/preproc/train*" \
    --eval_data_paths "$GCS_PATH/preproc/eval*" \
    --eval_set_size 19 \
    --output_path output/
```

## Appendix: image sources

The source information for the 'hugs/no-hugs' images is here: gs://oscon-tf-workshop-materials/transfer_learning/hugs_photos_sources.csv.
