# CNN Regression

This folder contains the code produced during the experiment with baseline U-net for a project "Quality control for deep learning-based image segmentation for reliable integration into medical workflows" performed by Elena Williams, Sebastian Niehaus, Janis Reinelt, Alberto Merola, Paul Glad Mihai,  Nico Scherf, Ingo Röder, Maria del C. Valdés Hernández.


### Utils

The most useful functions used for data loading, preprocessing and augmnetation were placed in the script file "utils".


### Gener 

Generator was used to load and preprocess a batch of input data and dynamically feed it into the training process.


### Train

Train files were used to intiate the training process with input of image-prediction (IP) pair, uncertainty-prediction (PE) pair, error-prediction (EM) pair.


### CNN Regression

This file contains the architecture of a CNN regression model.

### Evaluation 

Evaluation 1 was used to assess the performance of the models trained with an input of image-prediction (IP) pair and uncertainty-prediction (PE) pair. Evaluation 2 was used to perform analysis of the models trained with error-prediction (EM) pair. Evaluation notebook shows the mean QC performance for VS measure and Dice prediction model.


### res and training_hist folders

These folders contain the results of the evaluation as well as the training history for every fold.

### Dice Prediction 

This notebook was used to evaluate the training results, test the evaluation pipeline and visualise the training history.