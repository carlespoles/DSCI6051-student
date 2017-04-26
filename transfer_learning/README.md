Food images classification: Image-based transfer learning on Cloud ML
--------------------------------------------------

To run this example, first follow instructions for [setting up your environment](https://cloud.google.com/ml/docs/how-tos/getting-set-up).

It uses Apache Beam (running on Cloud Dataflow) and PIL to preprocess the images into embeddings, so make sure to install the required packages:
```
pip install -r requirements.txt
```

Then, you may follow the instructions in `tasty_images.sh`.

This directory contains an of transfer learning using the "Inception V3" image classification model.

This example shows how to use Cloud Dataflow (Apache Beam) to do image preprocessing, then train and serve a model on Cloud ML. It supports distributed training on Cloud ML. It is based on the [example here](https://cloud.google.com/blog/big-data/2016/12/how-to-classify-images-with-tensorflow-using-google-cloud-machine-learning-and-cloud-dataflow) and [adapting the code from this repository](https://github.com/amygdala/tensorflow-workshop/tree/master/workshop_sections/transfer_learning)<sup>1</sup> to this project needs. However, some API are deprecated from<sup>1</sup>, and I have refactored them using the most [recent example from GoogleCloudPlatform](https://github.com/GoogleCloudPlatform/cloudml-samples/tree/master/flowers).

It can be easily modified and adapted for any image classification problems with different images using "Inception V3". A prediction web server it's also included that demos how to use the Cloud ML API for prediction once your trained model is serving.

Detailed [instructions on how to run this project are found here](Instructions.md).
