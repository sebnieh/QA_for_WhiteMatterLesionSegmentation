# Data Preparation

This folder contains the code produced during the stage of data preparation for a project "Quality control for deep learning-based image segmentation for reliable integration into medical workflows" performed by Elena Williams, Sebastian Niehaus, Janis Reinelt, Alberto Merola, Paul Glad Mihai,  Nico Scherf, Ingo Röder, Maria del C. Valdés Hernández.


### Utils

The most useful functions used for data loading and preprocessing were stored in the script file "utils".


### MATLAB Scripts 

MATLAB scripts SS_MS17.m and SS_MICCAI17.m were used to load the MRI data which contains the skulls and perform brain extraction.


### Main Augmentation

Main Augmentation script includes functions to perform spatial transformations as well as noise application.

### Data Preparation notebook

In the data prepration notebook, the data was explored, the loading of every database was tested. In addition we averaged segmentation maps, calculated the distance between the brain volume percentage and WMH percentage.

### Test augmentation, data split

In the this notebook, I tested that the augmentation functions work well on all of the data and splitted the data into the train, validation and test sets.


### WML_dist

This CSV file includes distance values calculated in the data preparation notebook using the formula provided by Dr Maria del C. Valdés Hernández
