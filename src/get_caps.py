
from datetime import datetime
import numpy as np
import paths
from pathlib import Path
import nibabel as nib
import matplotlib.pyplot as plt
import utils
import pandas as pd
from sklearn.cluster import KMeans
from nilearn import plotting

group_path = paths.sample_derivatives / "non_preterm"
T = 15  # percentile threshold


gm_mask_path = paths.ext40GreyMatterMask
seed_mask_path = paths.ext40PosteriorCingulateGyrusMask
gm_mask = nib.load(gm_mask_path)
seed_mask = nib.load(seed_mask_path)
subj_vols_dirs = sorted(group_path.glob("sub-*/vols"))


retained_frames = []
for vol_dir in subj_vols_dirs:
    # load each 3d frames and apply gm mask to it, returns a list of tuples (frame_time, frame_vector)
    flat_vols = utils.get_masked_frames(vol_dir, gm_mask)
    seed_timecourse = utils.get_seed_timecourse(flat_vols, seed_mask, zscore=True)
    l, h = utils.get_percentile_thresholds(seed_timecourse, T)
    for frame_time, frame in enumerate(flat_vols):
        seed_activity = seed_timecourse[frame_time]
        activity_type = None
        if seed_activity < l:
            activity_type = "low"
        elif seed_activity > h:
            activity_type = "high"
        if activity_type:
            retained_frames.append((vol_dir.parent.name, frame_time, activity_type, frame))

    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Processed {vol_dir.parent.name} ({len(flat_vols)} vols), retained {len(retained_frames)} frames so far")

retained_frames_df = pd.DataFrame(retained_frames, columns=["subj_name", "frame_time", "type", "frame"]).set_index(["subj_name", "frame_time"])
del retained_frames

stacked_frames = np.stack(retained_frames_df['frame'].to_numpy())

kmeans = KMeans(n_clusters=5, random_state=0)
retained_frames_df['cluster'] = kmeans.fit_predict(stacked_frames)

CAPs = retained_frames_df.groupby('cluster')['frame'].apply(lambda x: np.mean(np.stack(x), axis=0))
CAPs = CAPs.apply(lambda x: (x - x.mean()) / x.std())  # zscore each CAP

# reshape each CAP to 3D and plot
for i, cap in CAPs.items():
    cap_3d = utils.unflatten_to_3d(cap, gm_mask)
    plotting.plot_stat_map(
        cap_3d,
        display_mode="ortho",
        cut_coords=None,
        colorbar=True,
        vmax=5,
        bg_img=paths.ext40Template,
        black_bg=False,
        cmap="RdBu_r",
    )

# in the end i want
# 1. easy access to retained_frames globally (to perform clustering on them)
# 2. a mapping between each retained frame and its subject, time index and it being either a high or low activation at seed frame

# for each subject,
#     get vols/ path,
#     load all of its frames, GM mask them,
#     compute z-scored mean activity at seed := s for each frame
#     compute T and 100 - T percentiles l,h
#     for each frame f,
#       if s < l
#           store subj_id, f_time, "low" and frame vector in dataframe
#       elif s > h:
#           store subj_id, f_time, "high" and frame vector in dataframe
#     append to global dataframe with
#     as index subj_id, time of the frame
#     and as columns "retain_reason" and the frame vector

# compute kmeans on df[frames]
# append to each row of df the cluster it belongs to

# for each group in df.groupby(cluster):
#   average, zscore, reshape to 3D, plot