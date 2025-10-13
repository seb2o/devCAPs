%% This is an example script to run the CAPs analyses without the GUI
% In this script, we assume one population of subjects only

addpath(genpath(pwd));

% ---- USER SETTINGS ----
% Threshold above which to select frames
T = 15;

% Selection mode ('Threshold' or 'Percentage')
SelMode = 'Percentage';

% Contains the information, for each seed (each row), about whether to
% retain activation (1 0) or deactivation (0 1) time points
SignMatrix = [1,0];

% Number of clusters to use
K_opt=5;

% run consensus clustering ?
run_consensus = false  % true or false

% folder where to find the data (in BIDS derivatives format)
derivatives_folder = '/home/boo/capslifespan/data/derivatives/';

% group folder in derivatives where to fetch subjects
group_name = 'non_preterm';

% optional custom prefix for output folder
custom_prefix = '';


derivatives_folder = string(derivatives_folder);
group_name         = string(group_name);
custom_prefix      = string(custom_prefix);
SelMode            = string(SelMode);

in_dir_path = fullfile(derivatives_folder, group_name);


%% 1. Loading the data files
t_start = datetime('now');
fprintf('>>> Start: %s\n', datestr(t_start, 'yyyy-mm-dd HH:MM:SS.FFF'));
tic;

% Data: cell array, each cell of size n_TP x n_masked_voxels
TC = TCGM(in_dir_path);
disp(['Shape of TC cell array: ', num2str(size(TC,1)), ' subjects, each with [', ...])
      num2str(size(TC{1},1)), ' time points x ', num2str(size(TC{1},2)), ' voxels]']);

if isequal(SignMatrix, [1, 0])
    sign_str = "pos";
else
    sign_str = "neg";
end

if ~strcmp(custom_prefix, "")
    custom_prefix = [custom_prefix "_"];
end

folder_name = custom_prefix + group_name + ...
    "_CAPS_k-" + string(K_opt) + ...
    "_t" + string(SelMode(1)) + "-" + string(T) + ...
    "_activation-" + sign_str + ...
    "_n-" + string(numel(TC));

out_dir_path = fullfile(char(derivatives_folder), char(folder_name));

fprintf('Output directory path: %s\n', out_dir_path);
out_dir = fullfile(out_dir_path);
if ~exist(out_dir, 'dir'); mkdir(out_dir); end


% Mask: n_voxels x 1 logical vector
V = spm_vol('/home/boo/capslifespan/data/templates/extdhcp40wkGreyMatterLowres_mask.nii');          % Read header
mask_data = spm_read_vols(V);          % Load the 3D data
mask = (mask_data(:) > 0);

% Seed: a n_masked_voxels x n_seed logical vector with seed information
Vseed = spm_vol('/home/boo/capslifespan/data/templates/extdhcp40wkPosteriorCingulateGyrusLowres_mask.nii');
seed_img = spm_read_vols(Vseed);           % 3D
seed_vec = seed_img(:) > 0;                % vectorize
Seed     = seed_vec(mask);                  % restrict to GM masked voxels
% If you had multiple seeds, concatenate as extra columns.
% e.g., Seed = [seed1_vec(mask) seed2_vec(mask)];


% Header: the header (obtained by spm_vol) of one NIFTI file with proper
% data dimension and .mat information
volume_files = dir('/home/boo/capslifespan/data/sample_derivatives/sub-*/vols/*3D_1.nii');
if ~isempty(volume_files)
    brain_info = spm_vol(fullfile(volume_files(1).folder, volume_files(1).name));
else
    error('No NIfTI files found');
end


% Framewise displacement: a n_TP x n_subjs matrix with framewise
% displacement information
FD = zeros(size(TC{1},1), numel(TC));





%% 2. Specifying the main parameters



% Threshold of FD above which to scrub out the frame and also the t-1 and
% t+1 frames (if you want another scrubbing setting, directly edit the
% code)
Tmot = 0.5;

% Type of used seed information: select between 'Average','Union' or
% 'Intersection' (only useful for multiple seeds)
SeedType = 'Average';



% Percentage of positive-valued voxels to retain for clustering
Pp = 100;

% Percentage of negative-valued voxels to retain for clustering
Pn = 100;

% Number of repetitions of the K-means clustering algorithm
n_rep = 50;

% Percentage of frames to use in each fold of consensus clustering
Pcc = 80;

% Number of folds we run consensus clustering for
N = 50;



%% 3. Selecting the frames to analyse    

% Xon will contain the retained frames, and Indices will tag the time
% points associated to these frames, for each subject (it contains a
% subfield for retained frames and a subfield for scrubbed frames)
[Xon,~,Indices] = CAP_find_activity(TC,Seed,T,FD,Tmot,SelMode,SeedType,SignMatrix);

size(Xon)

%% 3.1 Information on NaNs
n_subjects = numel(Xon);

total_vox      = zeros(n_subjects,1);
nan_voxels     = zeros(n_subjects,1);
pct_nan_vox    = zeros(n_subjects,1);

total_frames   = zeros(n_subjects,1);
nan_frames     = zeros(n_subjects,1);
pct_nan_frames = zeros(n_subjects,1);

for s = 1:n_subjects
    subj_data = Xon{s};
    if isempty(subj_data)
        total_vox(s)      = NaN;
        nan_voxels(s)     = NaN;
        pct_nan_vox(s)    = NaN;
        total_frames(s)   = NaN;
        nan_frames(s)     = NaN;
        pct_nan_frames(s) = NaN;
        continue;
    end
    
    % voxels
    total_vox(s)   = size(subj_data,1);
    nan_voxels(s)  = sum(any(isnan(subj_data),2));
    pct_nan_vox(s) = (nan_voxels(s) / total_vox(s)) * 100;
    
    % frames
    total_frames(s)   = size(subj_data,2);
    nan_frames(s)     = sum(any(isnan(subj_data),1));
    pct_nan_frames(s) = (nan_frames(s) / total_frames(s)) * 100;
end

B = table((1:n_subjects)', total_vox, nan_voxels, pct_nan_vox, ...
          total_frames, nan_frames, pct_nan_frames, ...
          'VariableNames', {'Subject','TotalVoxels','NaNVoxels','PercentNaNVoxels', ...
                            'TotalFrames','NaNFrames','PercentNaNFrames'});

first_save_path = fullfile(out_dir_path, 'variables_1st.mat')
save(first_save_path);

  
%% 3.5 Replace NaNs with zeros in Xon (keep frames)
nan_voxel_total = 0;  % counts how many voxels were replaced

for s = 1:numel(Xon)
    if isempty(Xon{s}), continue; end
    % Xon{s} is [n_voxels x n_selected_frames] for subject s
    
    nan_mask = isnan(Xon{s});          % logical mask of NaNs
    n_nan_voxels = nnz(nan_mask);      % number of NaNs in this subject
    if n_nan_voxels > 0
        Xon{s}(nan_mask) = 0;          % replace all NaNs with 0
        nan_voxel_total = nan_voxel_total + n_nan_voxels;
    end
end

fprintf('✅ Replaced %d NaN voxels with zeros across all subjects.\n', nan_voxel_total);
    

%% 4. Consensus clustering (if wished to determine the optimum K)
if run_consensus
% This specifies the range of values over which to perform consensus
% clustering: if you want to run parallel consensus clustering processes,
% you should feed in different ranges to each call of the function
    K_range = 3:15;

% Have each of these run in a separate process on the server =)
    [Consensus] = CAP_ConsensusClustering(Xon,K_range,'items',Pcc/100,N,'correlation');

% Calculates the quality metrics
    [~,PAC] = ComputeClusteringQuality(Consensus,[]);

% Qual should be inspected to determine the best cluster number(s)

end


%% Sanity check right before clustering
Xtmp = cell2mat(Xon);
fprintf('Frames after cleanup: %d, voxels: %d, anyNaN left: %d\n', ...
        size(Xtmp,2), size(Xtmp,1), any(isnan(Xtmp(:))));


%% 5. Clustering into CAPs (on cleaned Xon)

XONn = cell2mat(Xon);   % [n_voxels x total_retained_frames_after_cleanup]
[CAP,~,~,idx] = Run_Clustering_Sim(XONn, K_opt, mask, brain_info, Pp, Pn, n_rep, [], SeedType);



%% 5.1. Save CAPs as NIfTI maps (one file per cluster)

% -----------------------

K = size(CAP, 1);                 % number of clusters
volSize = brain_info.dim;         % [nx ny nz]
nVoxTot = prod(volSize);

% defensive checks
mask = mask(:) > 0;               % ensure logical column
if numel(mask) ~= nVoxTot
    error('Mask length (%d) does not match template voxel count (%d).', ...
           numel(mask), nVoxTot);
end
if size(CAP, 2) ~= sum(mask)
    error('CAP size (GM_size=%d) does not match number of true voxels in mask (%d).', ...
           size(CAP,2), sum(mask));
end

% Write one 3D NIfTI per CAP (raw and z-scored)
for k = 1:K
    % ----- RAW CAP -----
    flatVol_raw = zeros(nVoxTot, 1, 'single');
    flatVol_raw(mask) = single(CAP(k, :));
    vol3D_raw = reshape(flatVol_raw, volSize);

    Vout_raw        = brain_info;
    Vout_raw.fname  = fullfile(out_dir, sprintf('CAP_%02d_raw.nii', k));
    Vout_raw.descrip= sprintf('Raw CAP #%d (K=%d)', k, K);
    if isfield(Vout_raw,'dt'); Vout_raw.dt = [16 0]; end
    spm_write_vol(Vout_raw, vol3D_raw);

    % ----- Z-SCORED CAP -----
    cap_vals = CAP(k, :);
    cap_z = (cap_vals - mean(cap_vals)) ./ std(cap_vals);   % z-score

    flatVol_z = zeros(nVoxTot, 1, 'single');
    flatVol_z(mask) = single(cap_z);
    vol3D_z = reshape(flatVol_z, volSize);

    Vout_z        = brain_info;
    Vout_z.fname  = fullfile(out_dir, sprintf('CAP_%02d_z.nii', k));
    Vout_z.descrip= sprintf('Z-scored CAP #%d (K=%d)', k, K);
    if isfield(Vout_z,'dt'); Vout_z.dt = [16 0]; end
    spm_write_vol(Vout_z, vol3D_z);
end

fprintf('Saved %d raw and z-scored CAP maps to: %s\n', K, out_dir);


%% 6. Computing metrics

% Sanity check
fprintf('idx length = %d; selected frames = %d\n', numel(idx), nnz(Indices.kept.active));

% The TR of your data in seconds
TR = 0.392;

[ExpressionMap,Counts,Entries,Avg_Duration,Duration,TransitionProbabilities,...
    From_Baseline,To_Baseline,Baseline_resilience,Resilience,Betweenness,...
    InDegree,OutDegree,SubjectEntries] = Compute_Metrics_simpler(idx,...
    Indices.kept.active,Indices.scrubbedandactive,K_opt,TR);

t_end   = datetime('now');
elapsed = toc;
fprintf('>>> End:   %s\n', datestr(t_end,   'yyyy-mm-dd HH:MM:SS.FFF'));
fprintf('>>> Elapsed: %.2f seconds (%.2f minutes)\n', elapsed, elapsed/60);

second_save_path = fullfile(out_dir_path, 'variables_2nd.mat')
save(second_save_path);


%% =======================
% 6.5. Clean CAPs variables + group-level summary
% =======================

% 1. Overall Info
OverallInfo = struct();
OverallInfo.NumberSubjects   = size(Xon,2);
OverallInfo.NumberTimePoints = size(XONn,2);   % total retained frames across subjects
OverallInfo.NumberVoxels     = size(CAP,2);
OverallInfo.NumberClusters   = size(CAP,1);
OverallInfo.TR               = TR;
OverallInfo.MaskVoxels       = nnz(mask);

% 2. Outputs: CAPs & Metrics
Outputs = struct();

% KMeans clustering results
Outputs.KMeansClustering.CoActivationPatterns        = CAP;
Outputs.KMeansClustering.AssignmentsToCAPs           = idx;
Outputs.KMeansClustering.CoActivationPatternsZScored = (CAP - mean(CAP,2)) ./ std(CAP,0,2); % z-scored

% Metrics
Outputs.Metrics.AverageExpressionDuration    = Avg_Duration;
Outputs.Metrics.AllExpressionDurations       = Duration;
Outputs.Metrics.Occurrences                  = Counts;
Outputs.Metrics.TransitionProbabilities      = TransitionProbabilities;
Outputs.Metrics.CAPEntriesFromBaseline       = From_Baseline;
Outputs.Metrics.CAPExitsToBaseline           = To_Baseline;
Outputs.Metrics.CAPResilience                = Resilience;
Outputs.Metrics.BaselineResilience           = Baseline_resilience;
Outputs.Metrics.BetweennessCentrality        = Betweenness;
Outputs.Metrics.CAPInDegree                  = InDegree;
Outputs.Metrics.CAPOutDegree                 = OutDegree;
Outputs.Metrics.SubjectCounts                = SubjectEntries;

% 3. Save clean .mat file
if ~exist(out_dir, 'dir'), mkdir(out_dir); end

out_mat = fullfile(out_dir, 'variables_clean_10.mat');
save(out_mat, 'OverallInfo', 'Outputs', '-v7.3');
fprintf('✅ Clean .mat file saved: %s\n', out_mat);

% 4. Write group-level metrics summary (TXT)

nCAP      = size(CAP,1);
nSubjects = size(Avg_Duration,1);

% Compute group-level metrics (ignore NaNs)
nCAP      = size(CAP,1);
nSubjects = size(Avg_Duration,1);

AvgDur_group  = mean(Avg_Duration, 1, 'omitnan');      
Occ_group     = mean(Counts.raw.state, 1, 'omitnan');  % 1 x nCAP  
FromBase      = mean(From_Baseline, 1, 'omitnan');      
ToBase        = mean(To_Baseline, 1, 'omitnan');        
ResilienceG   = mean(Resilience, 1, 'omitnan');         
BaselineRes   = mean(Baseline_resilience, 'omitnan');   % scalar
BetwG         = mean(Betweenness, 1, 'omitnan');        
InDegreeG     = mean(InDegree, 1, 'omitnan');           
OutDegreeG    = mean(OutDegree, 1, 'omitnan');          

TP_group = mean(TransitionProbabilities, 3, 'omitnan');  % nCAP x nCAP

% TXT output
txt_file = fullfile(out_dir,'CAP_group_summary.txt');
fid = fopen(txt_file,'w');

fprintf(fid, '=== Group-level CAP metrics summary ===\n\n');
fprintf(fid, 'Number of CAPs: %d\nNumber of subjects: %d\n\n', nCAP, nSubjects);

for k = 1:nCAP
    fprintf(fid, 'CAP %d:\n', k);
    fprintf(fid, '  Avg Duration (frames): %.2f\n', AvgDur_group(k));
    fprintf(fid, '  Occurrences per subject: %.2f\n', Occ_group(k));
    fprintf(fid, '  From Baseline: %.2f\n', FromBase(k));
    fprintf(fid, '  To Baseline: %.2f\n', ToBase(k));
    fprintf(fid, '  Resilience: %.2f\n', ResilienceG(k));
    fprintf(fid, '  Betweenness Centrality: %.2f\n', BetwG(k));
    fprintf(fid, '  InDegree: %.2f\n', InDegreeG(k));
    fprintf(fid, '  OutDegree: %.2f\n\n', OutDegreeG(k));
end

% Add Baseline Resilience as a separate scalar
fprintf(fid, 'Baseline Resilience (group-level scalar): %.2f\n\n', BaselineRes);

fprintf(fid, '=== Group-level Transition Probability Matrix (CAP x CAP) ===\n');
for i = 1:nCAP
    for j = 1:nCAP
        fprintf(fid, '%.3f\t', TP_group(i,j));
    end
    fprintf(fid,'\n');
end

fclose(fid);
fprintf('✅ Group-level summary saved to: %s\n', txt_file);

