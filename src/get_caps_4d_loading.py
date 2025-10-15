import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from time import perf_counter

import nibabel as nib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

import paths
import show_caps
import utils


def main(
        group_path,
        t=15,
        threshold_type='percentage',
        n_clusters=4,
        n_inits=50,
        sel_mode='pos',
        cluster_dist='euclidean',
        load_retained_frames_df=False,
        recompute_clusters=True
):

    if not load_retained_frames_df and not recompute_clusters:
        raise ValueError("If not loading retained_frames_df, must recompute clusters")

    gm_mask_path = paths.ext40GreyMatterMask
    seed_mask_path = paths.ext40PosteriorCingulateGyrusMask
    gm_mask = nib.load(gm_mask_path)
    seed_mask = nib.load(seed_mask_path)
    subj_4dbolds_paths = sorted(group_path.glob("sub-*/ses-*/func/*bold.nii.gz"))
    n_subjs = len(subj_4dbolds_paths)

    expname = (
           f"dist-{cluster_dist}"
           f"_ttype-{threshold_type}"
           f"_tvalue-{t}"
           f"_k-{n_clusters}"
           f"_ninits-{n_inits}"
           f"_activation-{sel_mode}"
           f"_n-{n_subjs}"
    )
    savedir = group_path / expname
    savedir.mkdir(exist_ok=True)
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Starting CAP analysis from {group_path}, saving to {savedir}")

    # load one 3d bold to get affine, header, shape info
    sample_fourd = nib.load(subj_4dbolds_paths[0])
    sample_volume = sample_fourd.slicer[..., 0]
    del sample_fourd

    if not load_retained_frames_df:
        retained_frames = []
        times = []
        start_time = perf_counter()
        with ThreadPoolExecutor(max_workers=max(os.cpu_count(), 8)) as ex:

            futures = [
                ex.submit(utils.extract_subject_frames, bold_path, gm_mask, seed_mask, t, sel_mode)
                for bold_path in subj_4dbolds_paths
            ]

            for fut in as_completed(futures):
                res = fut.result()
                times.append(res["load_time"])
                retained_frames.extend(res["retained"])
                now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                print(
                    f"[{now}] Processed {res['subj_name']}, {res['n_vols']} vols of {res['n_voxels']} voxels each in {res['load_time']:.4f}s, retained {len(retained_frames)} frames so far"
                )

        end_time = perf_counter()
        total_time = end_time - start_time

        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Average time to load and mask a subject: {np.mean(times):.4f} seconds. Total time: {total_time:.2f} seconds to process {len(subj_4dbolds_paths)} subjects")

        retained_frames_df = pd.DataFrame(retained_frames, columns=["subj_name", "frame_time", "type", "frame"]).set_index(["subj_name", "frame_time"])
        del retained_frames
    else:
        retained_frames_df = pd.read_pickle(savedir / "retained_frames.pkl")
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Loaded retained_frames_df from {savedir / 'retained_frames.pkl'}, shape: {retained_frames_df.shape}")

    if recompute_clusters:

        stacked_frames = np.stack(retained_frames_df['frame'].to_numpy())

        # zscore samples to approximate correlation distance with euclidean
        # this is important for kmeans to work well
        zscored_stacked_frames = (stacked_frames - stacked_frames.mean(axis=1, keepdims=True)) / stacked_frames.std(axis=1, keepdims=True)

        kmeans = KMeans(
            n_clusters=n_clusters,
            random_state=0,
            n_init=n_inits,
        )


        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Starting Clustering")


        retained_frames_df['cluster'] = kmeans.fit_predict(zscored_stacked_frames)


        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Finished Clustering")


        clusters_value_counts = retained_frames_df['cluster'].value_counts()
        print(f"Cluster sizes: {clusters_value_counts.to_dict()}")
        cluster_order = clusters_value_counts.index
        cluster_map = {old: new for new, old in enumerate(cluster_order)}
        retained_frames_df['cluster'] = retained_frames_df['cluster'].map(cluster_map)


    CAPs = retained_frames_df.groupby('cluster')['frame'].apply(lambda x: np.mean(np.stack(x), axis=0))

    # reshape each CAP to 3D and save
    n = CAPs.shape[0]


    # Save zscored .nii files

    for i, cap in CAPs.items():
        cap_3d = utils.unflatten_to_3d_only_gm(cap, gm_mask, sample_volume,  zscore=True)
        nib.save(cap_3d, savedir / f"CAP_{i+1:02d}_z.nii")

    # Save analysed frames df with their clusters

    retained_frames_df.to_pickle(savedir / "retained_frames.pkl")

    # Save without the frame data to save space when only want to analyze clusters
    retained_frames_df.drop(columns="frame").to_pickle(savedir / "frames_clustering.pkl")

    # Save cluster sizes
    retained_frames_df['cluster'].value_counts().to_pickle(savedir / "cluster_sizes.pkl")

    # Save png views for each individual CAP, global overview, and detailed overview
    show_caps.plot_caps(
        folder_path=savedir,
        fig_title=f"CAPs in {group_path.name} ({n} total)",
        save_path=savedir / "CAPs_overview.png"
    )

    return savedir

if __name__ == "__main__":
    main(
        group_path=paths.sample_derivatives,
        t=15,
        threshold_type='percentage',
        n_clusters=4,
        n_inits=50,
        sel_mode='pos',
        cluster_dist='euclidean',
        load_retained_frames_df=False,
        recompute_clusters=True
    )
