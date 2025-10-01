function all_data = TCGM()

% ----------------------------
% SETTINGS
% ----------------------------
main_folder = '/home/boo/capslifespan/data/sample_derivatives';      % Folder containing subject folders
data_subfolder_name = 'vols';        % Name of subfolder inside each subject folder
mask_volume = spm_vol('/home/boo/capslifespan/data/templates/extdhcp40wkGreyMatterLowres_mask.nii')
mask_data = spm_read_vols(mask_volume);          % Load the 3D data
mask_vector = (mask_data(:) > 0);   % 1 for mask voxels, 0 otherwise

nGMvoxels = sum(mask_vector);
disp(['Using GM mask with ', num2str(nGMvoxels), ' voxels']);

% ----------------------------
% FIND SUBJECT FOLDERS
% ----------------------------
subject_folders = dir(main_folder);
subject_folders = subject_folders([subject_folders.isdir] & ...
                                  ~startsWith({subject_folders.name}, '.'));

all_data = cell(length(subject_folders), 1); % Preallocate

% ----------------------------
% LOOP OVER SUBJECTS
% ----------------------------
for s = 1:length(subject_folders)
    subj_path = fullfile(main_folder, subject_folders(s).name);
    data_folder = fullfile(subj_path, data_subfolder_name);
    
    if ~isfolder(data_folder)
        warning(['Data folder not found for subject: ', subject_folders(s).name]);
        continue;
    end
    
    % Find NIfTI files
    files = dir(fullfile(data_folder, '*.nii'));
    if isempty(files)
        warning(['No NIfTI files found for subject: ', subject_folders(s).name]);
        continue;
    end
    
    % Sort filenames naturally
    file_names = {files.name};
    sorted_files = natural_sort(file_names);  % Requires natsort.m
    full_paths = fullfile(data_folder, sorted_files);
    
    % Preallocate for this subject
    nTimepoints = length(full_paths);
    data_2d = zeros(nTimepoints, nGMvoxels);
    
    % Load & mask
    for t = 1:nTimepoints
        vol = spm_read_vols(spm_vol(full_paths{t}));
        vol_vector = vol(:);
        data_2d(t, :) = vol_vector(mask_vector);
    end
    
    % Store in cell array
    all_data{s} = data_2d;
    
    disp(['Loaded ', subject_folders(s).name, ...
          ': ', num2str(nTimepoints), ' × ', num2str(nGMvoxels), ' GM voxels']);
end

all_data = all_data';

disp('✅ Data extraction complete');
end