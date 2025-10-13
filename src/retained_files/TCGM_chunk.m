function all_data = TCGM_chunk()

% ----------------------------
% SETTINGS
% ----------------------------
main_folder = '/media/RCPNAS/Data/asalcedo/preproc_nii/3D_vols/';      % Folder containing subject folders
data_subfolder_name = 'vols';        % Name of subfolder inside each subject folder
mask_vector = load('/media/RCPNAS/Data/asalcedo/GM_masks/mask_vector_nodeep4_smooth.mat').mask_vector; % Your GM mask vector (logical)
mask_vector = logical(mask_vector);

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
    
    % ==== NEW: Crop to selected frames ====
    frame_start = 1;      % set start
    frame_end   = 760;    % set end
    frame_end = min(frame_end, size(data_2d,1)); % safety check
    data_2d = data_2d(frame_start:frame_end, :);
    
    % Store in cell array
    all_data{s} = data_2d;
    
    disp(['Loaded ', subject_folders(s).name, ...
          ': ', num2str(nTimepoints), ' × ', num2str(nGMvoxels), ' GM voxels']);
end

all_data = all_data';

disp('✅ Data extraction complete');
end 