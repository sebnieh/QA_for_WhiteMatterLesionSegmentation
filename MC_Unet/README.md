# MCD U-net

This folder contains the code produced during the experiment with MC U-net for a project "Quality control for deep learning-based image segmentation for reliable integration into medical workflows" performed by Elena Williams, Sebastian Niehaus, Janis Reinelt, Alberto Merola, Paul Glad Mihai,  Nico Scherf, Ingo Röder, Maria del C. Valdés Hernández.


### Utils

The most useful functions used for data loading and preprocessing were placed in the script file "utils".

### Main Augmentation

Main Augmentation script includes functions to perform spatial transformations as well as noise application.

### Generator 

Generator was used to load and preprocess a batch of MRI data and dynamically feed it into the training process.

### Training 1 and 2

Training 1 and 2 files were used to intiate the training process with 2 folds running on GPU #1 and 3 folds running on GPU #2.

### Loss functions

Dice metric and loss functions were implemented using tensorflow operations.

### MCD Unet + Tfk instance

These files contain the modified U-net architecture with Dropout layers as well as the Instance Normalisation layer function.

### Evaluation notebook

In the evaluation notebook, you can see the analysis of the training history as well as the average performance of the network across 5-folds and examples of the predicted WMH. Additionally, we present how the extraction of the uncertainty was performed using an entropy formula and aggregation of uncertainty estimates using a voxel-wise sum measure.

### Prepare data

The scripts prepare_data_DR.py and prepare_test.py were used to save the data, particularly MRI data, segmentation predictions and uncertainty maps for further usage in CNN Regression experiments.

### ids and paths folders

ID numbers of data samples were defined in the data preparation. Due to the difference in the operating systems, the data samples used on the server did not match the order of the one loaded on the local computer. The paths were saved from the server to be able to load the test samples similarly on the local computer.

### res and training hist

These folders contain the results of the evaluation as well as the training history for every fold.
