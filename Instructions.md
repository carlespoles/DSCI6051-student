1 - EDA.
--------
Notebook `1-tasty-images-EDA.ipynb` is intended to explore data and perform some initial analysis to gain insights as well as present some visualizations.

2 - Getting the data.
---------------------
After EDA, it was decided that the best images are from `epicurious.com` as they have high social media scores and they are taken by professionals, so the good images will be curated from that site. Also, the worse images are coming from `food.com` as they have low social scores and they are taken by amateurs by their smartphones. Even though a given food recipe may high reviews, the associated photos can be of really bad quality, not appealing at all.

Also, most of the food recipes acquired by Wellio are sourced from `food.com` and `epicurious.com`.

We will get 10,000 images from each site, totaling a balanced data set of 20,000 images.

Images will be copied over a bucket in Google Cloud Storage as the storage in Datalab is ephemeral.
