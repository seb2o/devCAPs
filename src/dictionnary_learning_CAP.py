import os

import pandas as pd
import psutil
import sklearn
from matplotlib import pyplot as plt

import paths, utils, show_caps
import nibabel as nib
import numpy as np
from datetime import datetime


def main(
        group_path,
        t=15,
        sel_mode='pos',
        optional_path_prefix=None,
        load_retained_frames_df=False,
):
    pid = os.getpid()
    process = psutil.Process(pid)

    gm_mask_path = paths.ext40GreyMatterMask
    seed_mask_path = paths.ext40PosteriorCingulateGyrusMask
    gm_mask = nib.load(gm_mask_path)
    seed_mask = nib.load(seed_mask_path)
    subj_4dbolds_paths = sorted(group_path.glob("sub-*/ses-*/func/*bold.nii.gz"))
    n_subjs = len(subj_4dbolds_paths)

    if not optional_path_prefix:
        optional_path_prefix = ""
    elif not optional_path_prefix.endswith("_"):
        optional_path_prefix += "_"

    expname = (
        f"{optional_path_prefix}"
        f"DictionaryLearning"
        f"_ttype-percentage"      
        f"_tvalue-{t}"      
        f"_activation-{sel_mode}"       
        f"_n-{n_subjs}"
    )
    savedir = group_path / expname
    savedir.mkdir(exist_ok=True)
    print(
        f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] (PID {pid})"
        f" Starting Dictionnary Learning analysis from {group_path}, saving to {savedir}"
    )

    if load_retained_frames_df:
        retained_frames_df = pd.read_pickle(
            savedir / paths.retained_frames_wo_clusters_df_name
        )
    else:
        retained_frames_df = utils.get_frames(
            subj_4dbolds_paths=subj_4dbolds_paths,
            gm_mask=gm_mask,
            seed_mask=seed_mask,
            t=t,
            sel_mode=sel_mode,
            savedir=savedir,
            num_workers=2
        )

    utils.print_memstate(message="Before copying frames: ")

    stacked_frames = np.stack(retained_frames_df['frame'].to_numpy(copy=True))

    utils.print_memstate(message="After copying frames: ")

    del retained_frames_df

    utils.print_memstate(message="After removing frames: ")


    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Starting Dictionary Learning fitting")

    dico = sklearn.decomposition.DictionaryLearning(
        n_components=4,
        alpha=1,
        max_iter=500,
        fit_algorithm='lars',
        transform_algorithm='omp',
        random_state=0,
    )

    assignments = dico.fit_transform(stacked_frames)
    comps = dico.components_

    for comp_id in range(comps.shape[0]):
        plt.plot(assignments[:, comp_id], label=f"Component {comp_id+1}")
    plt.xlabel("Frame index")
    plt.ylabel("Component activation")
    plt.title("Dictionary Learning Component Activations over Frames")
    plt.legend()
    plt.show()

    for comp_id in range(comps.shape[0]):
        plt.figure()
        plt.hist(comps[comp_id])
        plt.title(f"Histogram of Component {comp_id+1}")
        plt.show()

    del stacked_frames

    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Dictionary Learning fitting done")



    sample_volume = utils.get_sample_volume(subj_4dbolds_paths[0])
    n_comps = len(comps)

    for comp_id, comp in enumerate(comps):
        comp3d = utils.unflatten_to_3d_only_gm(
            comp,
            gm_mask=gm_mask,
            sample_volume=sample_volume,
            zscore=False
        )
        nib.save(comp3d, savedir / f"CAP_{comp_id+1:02d}_z.nii")

    show_caps.plot_caps(
        folder_path=savedir,
        fig_title=f"CAPs in {group_path.name} ({n_comps} total)",
        save_path=savedir / "CAPs_overview.png"
    )

    return savedir

if __name__ == "__main__":
    main(
        group_path=paths.sample_derivatives,
        t=15,
        sel_mode='pos',
        optional_path_prefix="",
        load_retained_frames_df=False
    )
