Food images: Image-based transfer learning on Cloud ML
--------------------------------------------------

To run this example, first follow instructions for [setting up your environment](https://cloud.google.com/ml/docs/how-tos/getting-set-up).

Also, we use Apache Beam (running on Cloud Dataflow) and PIL to preprocess the images into embeddings, so make sure to install the required packages:
```
pip install -r requirements.txt
```

Then, you may follow the instructions in sample.sh.

This directory contains an of transfer learning using the "Inception V3" image classification model.

This example shows how to use Cloud Dataflow (Apache Beam) to do image preprocessing, then train and serve a model on Cloud ML. It supports distributed training on Cloud ML. It is based on the [example here<sup>1</sup>] [https://cloud.google.com/blog/big-data/2016/12/how-to-classify-images-with-tensorflow-using-google-cloud-machine-learning-and-cloud-dataflow], with some [additional modifications] (https://github.com/amygdala/tensorflow-workshop/tree/master/workshop_sections/transfer_learning) to adapt it to the food image set, and a prediction web server that demos how to use the Cloud ML API for prediction once your trained model is serving.

Note that above<sup>1</sup> has many libraries deprecated, and I have refactored them using the most recent example from GoogleCloudPlatform (https://github.com/GoogleCloudPlatform/cloudml-samples/tree/master/flowers)

It can be easily adapted for any image classification problem using "Inception V3".
