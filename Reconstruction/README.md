# Reconstruction 

This folder contains the code produced during the experiment with reconstruction network for a project "Quality control for deep learning-based image segmentation for reliable integration into medical workflows" performed by Elena Williams, Sebastian Niehaus, Janis Reinelt, Alberto Merola, Paul Glad Mihai,  Nico Scherf, Ingo Röder, Maria del C. Valdés Hernández.


### Utils

The most useful functions used for augmentation and preprocessing were placed into the script file "utils".

### Gener 

Generator was used to augment a batch of input data and dynamically feed it into the training process.

### Train 5cv

Training file was used to intiate the training process with 5-folds running on 2 GPUs.

### rec network + Tfk instance

These files contain the modified U-net architecture with ReLU activation function as well as the Instance Normalisation layer function.

### Evaluation 

Evaluation script was used to estimate SSIM index on test data across 5-folds. Evaluation notebook shows the mean QC performance for VS measure applied on uncertainty and error maps as well as performance of Dice prediction model.

### Prepare data

These scripts were used to save error maps produced with reconstructed FLAIR data for further usage in CNN Regression experiments.

### res and training hist

These folders contains the results of the evaluation as well as the training history for every fold.

### Reconstruction Evaluation 

This notebook was used to evaluate the training results, test the evaluation pipeline and visualise the training history.