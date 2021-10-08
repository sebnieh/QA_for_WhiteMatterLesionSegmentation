%
%/*******************************************************
% * Copyright (C) 2020-2025 Elena Williams elena.williams@aicura-medical.com
% * 
% * This file is part of Thesis Project «Comparative Analysis of Quality Control Methods for a CNN-Based White Matter %Hyperintensities Segmentation Scheme».
% * 
% * This project cannot be copied and/or distributed without the express
% * permission of Elena Williams 
% *******************************************************/

%Author: Elena Williams (100%)

% A pipeline for brain extraction

path_var = readtable('/Users/elena/Documents/data/train/lj1/path.txt', 'Delimiter',';');
path_var(1,:) = [];
path_var = table2array(path_var);

for i = 1:length(path_var)
    [filepath,name,ext] = fileparts(char(path_var(i,1))); % read in the data paths
    t1 = niftiread(char(path_var(i,1))); % read in Nifti file, T1
    info_t1 = niftiinfo(char(path_var(i,1))); % read in metadata

    flair = niftiread(char(path_var(i,2)));  % read in Nifti file, FLAIR
    info_flair = niftiinfo(char(path_var(i,2))); % read in metadata
    mask = niftiread(char(path_var(i,3))); % read in Nifti file, brain mask
    masked_t1 = bsxfun(@times, t1, cast(mask, 'like', t1)); % extract the brain
    masked_flair = bsxfun(@times, flair, cast(mask, 'like', flair)); % extract the brain
    id = num2str(sscanf(name,'patient%d_T1.nii'));
    name1 = ['lj1/',id, '/T1.nii'];
    niftiwrite(masked_t1, name1, info_t1, 'Compressed', true); % save T1
    name2 = ['lj1/',id, '/FLAIR.nii'];
    niftiwrite(masked_flair, name2, info_flair, 'Compressed', true) % save FLAIR
end