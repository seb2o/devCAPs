import gc
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
        n_comps=4,
        positive_code=False,
        alpha=1.0,
        subject_loading_n_workers=2,
):
    pid = os.getpid()

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
        f"DictLr"
        f"_Ncomps-{n_comps}"
        f"_PosCode-{positive_code}"
        f"_Alpha-{alpha}"
        f"_Tvalue-{t}"      
        f"_Act-{sel_mode}"
        f"_n-{n_subjs}"
    )

    savedir = (paths.results / group_path.relative_to(paths.data)) / expname
    savedir.mkdir(exist_ok=True, parents=True)

    print(
        f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] (PID {pid})"
        f" Starting Dictionnary Learning analysis from {group_path}, saving to {savedir}"
    )

    if load_retained_frames_df:
        frames = pd.read_pickle(
            savedir / paths.retained_frames_wo_clusters_df_name
        )['frame'].to_numpy()
    else:
        frames = utils.get_frames(
            subj_4dbolds_paths=subj_4dbolds_paths,
            gm_mask=gm_mask,
            seed_mask=seed_mask,
            t=t,
            sel_mode=sel_mode,
            savedir=savedir,
            num_workers=subject_loading_n_workers
        )
    utils.print_memstate(message="After getting frames: ")

    stacked_frames = np.stack(frames, axis=0)
    del frames
    gc.collect()
    utils.print_memstate(message="After stacking frames: ")



    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Starting Dictionary Learning fitting")

    dico = sklearn.decomposition.DictionaryLearning(
        n_components=n_comps,
        alpha=alpha,
        max_iter=500,
        fit_algorithm='cd',
        transform_algorithm='lasso_cd',
        random_state=0,
        positive_code=positive_code,
    )

    assignments = dico.fit_transform(stacked_frames)

    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Dictionary Learning fitting done")

    comps = dico.components_
    del stacked_frames
    gc.collect()
    utils.print_memstate(message="After Dictionary Learning fitting: ")


    # reorder components by mean value across all assignments
    mean_assignments = assignments.mean(axis=0)
    sorted_comp_indices = np.argsort(-mean_assignments)  # descending order
    comps = comps[sorted_comp_indices]
    assignments = assignments[:, sorted_comp_indices]


    # save components
    np.save(savedir / paths.dictcomps_npy_name, comps)
    retained_frames_df = pd.read_pickle(savedir / paths.retained_frames_wo_clusters_df_name)

    assignments_df = pd.DataFrame(
        assignments,
        index=retained_frames_df.index,
        columns=range(assignments.shape[1])
    )
    assignments_df.columns = pd.MultiIndex.from_product(
        [
            ["DictComp"],
            list(range(comps.shape[0]))
        ],
        names=[
            "block",
            "idx"
        ]
    )
    retained_frames_df.columns = pd.MultiIndex.from_product(
        [
            ["RetainedFrames"],
            retained_frames_df.columns
        ],
        names=[
            "block",
            "info"
        ]
    )
    frames_with_assignments_df = pd.concat([retained_frames_df, assignments_df], axis=1)

    frames_with_assignments_df.to_pickle(
        savedir / paths.comp_assignments_with_frames_df_name
    )
    frames_with_assignments_df.drop(columns=[("RetainedFrames", "frame")]).to_pickle(
        savedir / paths.comp_assignments_df_name
    )

    sample_volume = utils.get_sample_volume(subj_4dbolds_paths[0])
    n_comps = len(comps)
    vmax=0
    for comp_id, comp in enumerate(comps):
        comp3d = utils.unflatten_to_3d_only_gm(
            comp,
            gm_mask=gm_mask,
            sample_volume=sample_volume,
            zscore=True
        )
        if np.abs(comp3d.get_fdata()).max() > vmax: vmax = np.abs(comp3d.get_fdata()).max()
        nib.save(comp3d, savedir / f"DictComp_{comp_id+1:02d}_z.nii")

    show_caps.plot_caps(
        folder_path=savedir,
        fig_title=f"DictComps in {savedir.name} ({n_comps} total)",
        save_path=savedir / "DictComps_overview.png",
        caps_glob="DictComp_*_z.nii",
        vmax=vmax
    )

    return savedir

if __name__ == "__main__":

    main(
        group_path=paths.sample_derivatives,
        t=15,
        sel_mode='pos',
        optional_path_prefix="",
        load_retained_frames_df=False,
        n_comps=40,
        positive_code=True,
        alpha=2.0,
        subject_loading_n_workers=os.cpu_count()//4,
    )

    # main(
    #     group_path=paths.sample_derivatives,
    #     t=15,
    #     sel_mode='pos',
    #     optional_path_prefix="",
    #     load_retained_frames_df=False,
    #     n_comps=4,
    #     positive_code=True,
    #     alpha=1.0,
    #     subject_loading_n_workers=os.cpu_count() // 4,
    # )
    #
    # main(
    #     group_path=paths.sample_derivatives,
    #     t=15,
    #     sel_mode='pos',
    #     optional_path_prefix="",
    #     load_retained_frames_df=False,
    #     n_comps=4,
    #     positive_code=True,
    #     alpha=2.0,
    #     subject_loading_n_workers=os.cpu_count() // 4,
    # )
