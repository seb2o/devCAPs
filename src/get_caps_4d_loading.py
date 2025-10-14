
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
import show_caps


def main(group_path, T, expname, load_retained_frames_df=False):
    gm_mask_path = paths.ext40GreyMatterMask
    seed_mask_path = paths.ext40PosteriorCingulateGyrusMask
    gm_mask = nib.load(gm_mask_path)
    seed_mask = nib.load(seed_mask_path)
    savedir = group_path / expname
    savedir.mkdir(exist_ok=True)
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Starting CAP analysis from {group_path}, saving to {savedir}, T={T}")

    if not load_retained_frames_df:
        subj_4dbolds_paths = sorted(group_path.glob("sub-*/ses-*/func/*bold.nii.gz"))
        retained_frames = []
        times = []
        for bold_path in subj_4dbolds_paths:
            subj_name = bold_path.parent.parent.parent.name

            start = perf_counter()
            masked_timeserie = utils.get_masked_frames_4d(bold_path, gm_mask)
            end = perf_counter()
            times.append(end-start)
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Loaded and masked {subj_name} in {end-start:.4f} seconds")

            seed_timecourse = utils.get_seed_timecourse_from4d(masked_timeserie, seed_mask, zscore=True)

            l, h = utils.get_percentile_thresholds(seed_timecourse, T)

            for frame_time in range(masked_timeserie.shape[-1]):
                seed_activity = seed_timecourse[frame_time]
                activity_type = None
                if seed_activity < l:
                    #activity_type = "low"
                    pass # skipping low activity frames for now
                elif seed_activity > h:
                    activity_type = "high"
                    #pass
                if activity_type:
                    retained_frames.append((subj_name, frame_time, activity_type, masked_timeserie[..., frame_time].flatten()))

            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Processed {subj_name} ({masked_timeserie.shape[-1]} vols), retained {len(retained_frames)} frames so far")

        print(f"Average time to load and mask a subject: {np.mean(times):.4f} seconds")

        retained_frames_df = pd.DataFrame(retained_frames, columns=["subj_name", "frame_time", "type", "frame"]).set_index(["subj_name", "frame_time"])
        del retained_frames
    else:
        retained_frames_df = pd.read_pickle(savedir / "retained_frames.pkl")
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Loaded retained_frames_df from {savedir / 'retained_frames.pkl'}, shape: {retained_frames_df.shape}")

    stacked_frames = np.stack(retained_frames_df['frame'].to_numpy())

    # zscore samples to approximate correlation distance with euclidean
    # this is important for kmeans to work well
    stacked_frames = (stacked_frames - stacked_frames.mean(axis=1, keepdims=True)) / stacked_frames.std(axis=1, keepdims=True)

    kmeans = KMeans(
        n_clusters=5,
        random_state=0,
        n_init=10,
    )


    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Starting Clustering")


    retained_frames_df['cluster'] = kmeans.fit_predict(stacked_frames)


    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Finished Clustering")


    cluster_order = retained_frames_df['cluster'].value_counts(ascending=True).index
    cluster_map = {old: new for new, old in enumerate(cluster_order)}
    retained_frames_df['cluster'] = retained_frames_df['cluster'].map(cluster_map)


    CAPs = retained_frames_df.groupby('cluster')['frame'].apply(lambda x: np.mean(np.stack(x), axis=0))
    CAPs = CAPs.apply(lambda x: (x - x.mean()) / x.std())  # zscore each CAP

    # reshape each CAP to 3D and save
    n = CAPs.shape[0]


    for i, cap in CAPs.items():
        cap_3d = utils.unflatten_to_3d(cap, gm_mask)
        nib.save(cap_3d, savedir / f"CAP_{i+1:02d}_z.nii")

    #save retained_frames_df for further analysis
    retained_frames_df.to_pickle(savedir / "retained_frames.pkl")
    # save without the frame data to save space when only want to analyze clusters
    retained_frames_df.drop(columns="frame").to_pickle(savedir / "frames_clustering.pkl")

    # extract png view for caps
    show_caps.plot_caps(
        folder_path=savedir,
        fig_title=f"CAPs in {group_path.name} ({n} total)",
        save_path=savedir / "CAPs_overview.png"
    )

if __name__ == "__main__":
    gpath = paths.sample_derivatives
    t = 15
    expname="negative_caps_t_15"
    main(gpath, t, expname, load_retained_frames_df=True)
