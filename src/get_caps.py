
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
from time import perf_counter


def main(group_path, T, expname):
    gm_mask_path = paths.ext40GreyMatterMask
    seed_mask_path = paths.ext40PosteriorCingulateGyrusMask
    gm_mask = nib.load(gm_mask_path)
    seed_mask = nib.load(seed_mask_path)
    subj_vols_dirs = sorted(group_path.glob("sub-*/vols"))

    savedir = group_path / expname
    print(f"saving CAPs and df to {savedir}")
    savedir.mkdir(exist_ok=True)
    retained_frames = []
    for vol_dir in subj_vols_dirs:
        # load each 3d frames and apply gm mask to it, returns a list of tuples (frame_time, frame_vector)
        start = perf_counter()
        flat_vols = utils.get_masked_frames(vol_dir, gm_mask)
        end = perf_counter()
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Loaded and masked {vol_dir.parent.name} in {end-start:.4f} seconds")
        seed_timecourse = utils.get_seed_timecourse(flat_vols, seed_mask, zscore=True)
        l, h = utils.get_percentile_thresholds(seed_timecourse, T)
        for frame_time, frame in enumerate(flat_vols):
            seed_activity = seed_timecourse[frame_time]
            activity_type = None
            if seed_activity < l:
                # activity_type = "low"
                pass # skipping low activity frames for now
            elif seed_activity > h:
                activity_type = "high"
            if activity_type:
                retained_frames.append((vol_dir.parent.name, frame_time, activity_type, frame))

        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Processed {vol_dir.parent.name} ({len(flat_vols)} vols), retained {len(retained_frames)} frames so far")

    retained_frames_df = pd.DataFrame(retained_frames, columns=["subj_name", "frame_time", "type", "frame"]).set_index(["subj_name", "frame_time"])
    del retained_frames

    stacked_frames = np.stack(retained_frames_df['frame'].to_numpy())

    # zscore samples
    stacked_frames = (stacked_frames - stacked_frames.mean(axis=1, keepdims=True)) / stacked_frames.std(axis=1, keepdims=True)

    # todo use correlation as distance and more init
    kmeans = KMeans(
        n_clusters=5,
        random_state=0,
        n_init=300,
    )
    retained_frames_df['cluster'] = kmeans.fit_predict(stacked_frames)

    cluster_order = retained_frames_df['cluster'].value_counts().index
    cluster_map = {old: new for new, old in enumerate(cluster_order)}
    retained_frames_df['cluster'] = retained_frames_df['cluster'].map(cluster_map)


    CAPs = retained_frames_df.groupby('cluster')['frame'].apply(lambda x: np.mean(np.stack(x), axis=0))
    CAPs = CAPs.apply(lambda x: (x - x.mean()) / x.std())  # zscore each CAP

    # reshape each CAP to 3D and save
    for i, cap in CAPs.items():
        cap_3d = utils.unflatten_to_3d(cap, gm_mask)
        nib.save(cap_3d, savedir / f"CAP_{i+1}_z.nii")

    # save retained_frames_df for further analysis
    retained_frames_df.to_pickle(savedir / "retained_frames.pkl")

if __name__ == "__main__":
    gpath = paths.sample_derivatives / "non_preterm"
    t = 15
    expname="Pos_caps"
    main(gpath, t, expname)