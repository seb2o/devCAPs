import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from time import perf_counter
import gc
import nibabel as nib
import numpy as np
import pandas as pd
import sklearn
import cust_kmeans

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
        optional_path_prefix="",
        load_retained_frames_df=False,
        recompute_clusters=True
):

    pid = os.getpid()

    if not load_retained_frames_df and not recompute_clusters:
        raise ValueError("If not loading retained_frames_df, must recompute clusters")

    gm_mask_path = paths.ext40GreyMatterMask
    seed_mask_path = paths.ext40PosteriorCingulateGyrusMask
    gm_mask = nib.load(gm_mask_path)
    seed_mask = nib.load(seed_mask_path)
    subj_4dbolds_paths = sorted(group_path.glob("sub-*/ses-*/func/*bold.nii.gz"))
    n_subjs = len(subj_4dbolds_paths)

    expname = (
        f"{optional_path_prefix}"       
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
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] (PID {pid}) Starting CAP analysis from {group_path}, saving to {savedir}")

    # load one 3d bold to get affine, header, shape info
    sample_fourd = nib.load(subj_4dbolds_paths[0])
    sample_volume = sample_fourd.slicer[..., 0]
    del sample_fourd

    if not load_retained_frames_df:
        frames = utils.get_frames(
            subj_4dbolds_paths=subj_4dbolds_paths,
            gm_mask=gm_mask,
            seed_mask=seed_mask,
            t=t,
            sel_mode=sel_mode,
            savedir=savedir,
            num_workers=2
        )
    else:
        frames = pd.read_pickle(savedir / paths.retained_frames_wo_clusters_df_name)['frame'].to_numpy()
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Loaded frames from {savedir / paths.retained_frames_wo_clusters_df_name}")

    utils.print_memstate(message="After getting frames: ")

    stacked_frames = np.stack(frames, axis=0)
    del frames
    gc.collect()
    utils.print_memstate(message="After stacking frames: ")

    # zscore samples to approximate correlation distance with euclidean
    # this is important for kmeans to work well
    # zscored_stacked_frames = (stacked_frames - stacked_frames.mean(axis=1, keepdims=True)) / stacked_frames.std(axis=1, keepdims=True)

    # kmeans = sklearn.cluster.KMeans(
    #     n_clusters=n_clusters,
    #     random_state=0,
    #     n_init=n_inits,
    # )
    # retained_frames_df['cluster'] = kmeans.fit_predict(zscored_stacked_frames)

    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Starting Clustering")

    if cluster_dist=='euclidean':
        best_xtocentre = sklearn.cluster.KMeans(
            n_clusters=n_clusters,
            random_state=0,
            n_init=n_inits,
            max_iter=300,
            tol=1e-4
        ).fit_predict(stacked_frames)

    elif cluster_dist=="correlation":
       # stacked_frames -= stacked_frames.mean(axis=1, keepdims=True)
       # stacked_frames /= np.linalg.norm(stacked_frames, axis=1, keepdims=True, ord=2)
       # best_xtocentre = sklearn.cluster.KMeans(
       #     n_clusters=n_clusters,
       #     random_state=0,
       #     n_init=n_inits,
       #     max_iter=300,
       #     tol=1e-4
       # ).fit_predict(stacked_frames)

        best_xtocentre = cust_kmeans.kmeans_corr(
            X=stacked_frames,
            n_clusters=n_clusters ,
            n_inits=n_inits ,
            max_iter=300 ,
            tol=1e-4 ,
            random_state=0
        )


    else:
        best_centres, best_xtocentre, best_distances, best_inertia = cust_kmeans.kmeans_with_n_init_withDebugging(
            X=stacked_frames,
            nclusters=n_clusters,
            n_init=n_inits,
            delta=1e-4,
            maxiter=300,
            metric=cluster_dist,
            verbose=2,
        )

    utils.print_memstate("After clustering: ")

    del stacked_frames
    gc.collect()
    utils.print_memstate("After deleting stacked frames: ")


    retained_frames_df = pd.read_pickle(savedir / paths.retained_frames_wo_clusters_df_name)

    retained_frames_df['cluster'] = best_xtocentre

    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Finished Clustering")



    clusters_value_counts = retained_frames_df['cluster'].value_counts()
    print(f"Cluster sizes: {clusters_value_counts.to_dict()}")
    cluster_order = clusters_value_counts.index
    cluster_map = {old: new for new, old in enumerate(cluster_order)}
    retained_frames_df['cluster'] = retained_frames_df['cluster'].map(cluster_map)

    # Save analysed frames df with their clusters
    retained_frames_df.to_pickle(savedir / paths.retained_frames_df_name)

    # Save without the frame data to save space when only want to analyze clusters
    retained_frames_df.drop(columns="frame").to_pickle(savedir / paths.frames_clustering_df_name)

    # Save cluster sizes
    retained_frames_df['cluster'].value_counts().to_pickle(savedir / paths.cluster_sizes_df_name)

    CAPs = retained_frames_df.groupby('cluster')['frame'].apply(lambda x: np.mean(np.stack(x), axis=0))

    del retained_frames_df

    # reshape each CAP to 3D and save
    n = CAPs.shape[0]


    # Save zscored .nii files

    for i, cap in CAPs.items():
        cap_3d = utils.unflatten_to_3d_only_gm(cap, gm_mask, sample_volume,  zscore=True)
        nib.save(cap_3d, savedir / f"CAP_{i+1:02d}_z.nii")

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
        sel_mode='both',
        cluster_dist='correlation',
        optional_path_prefix="vectorizedCorr_kmeans_",
        load_retained_frames_df=True,
    )
