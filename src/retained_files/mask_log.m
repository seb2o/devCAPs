% Load the mask NIfTI
V = spm_vol('/home/menbuas/PycharmProjects/miplab/data/templates/extdhcp40wkGreyMatter_mask.nii.gz');          % Read header
mask_data = spm_read_vols(V);          % Load the 3D data

% Flatten into a column vector and make binary
mask_vector = (mask_data(:) > 0);   % 1 for mask voxels, 0 otherwise

% Check size
disp(['Mask vector size: ', num2str(length(mask_vector)), ' x 1']);

% Optionally save as .mat file
save('/home/menbuas/PycharmProjects/miplab/data/templates/extdhcp40wkGreyMatter_mask.mat', 'mask_vector');
