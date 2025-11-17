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
import spams

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
        n_iters=300,
        positive_atoms=True,
        n_inits=50
):

    if positive_code: raise NotImplementedError("SPAMS does not support positive coding only yet.")

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
        f"SPAMS_DictLr"
        f"_Ncomps-{n_comps}"
        f"_PosCode-{positive_code}"
        f"_PosAtoms-{positive_atoms}"
        f"_nIters-{n_iters}"
        f"_nInits-{n_inits}"
        f"_alpha-{alpha}"
        f"_t-{t}"
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

    reshaped_stacked_frames = np.asfortranarray(stacked_frames.T, dtype=np.double)
    del stacked_frames
    gc.collect()
    utils.print_memstate(message=f"After putting stacked frames to {reshaped_stacked_frames.dtype} fortran array: ")

    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Starting Dictionary Learning fitting")
    per_seed_results = []
    for init in range(n_inits):

        # dict_inits = np.asfortranarray(np.random.default_rng(seed=init).random(
        #     size=(reshaped_stacked_frames.shape[0], n_comps),
        #     dtype=np.double
        # ))

        # select k random frames as initial dictionary
        col_ids = np.random.default_rng(seed=init).choice(
            reshaped_stacked_frames.shape[1],
            size=n_comps,
            replace=False
        )
        dict_inits = reshaped_stacked_frames[:, col_ids]

        D = spams.trainDL(
            reshaped_stacked_frames,
            K=n_comps,
            D=dict_inits,
            mode=3,
            lambda1=alpha,
            numThreads=subject_loading_n_workers,
            batchsize=512,
            verbose=False,
            iter=n_iters,
            posD=positive_atoms,
            return_model=False
        ) # D shape (n_dimensions, n_components) (each column is a component)

        # print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Dictionary learned, now computing assignments")

        Alpha = spams.omp(
            reshaped_stacked_frames,
            D,
            L=alpha,
            numThreads=subject_loading_n_workers
        )
        #sum_{i=1}^n (1/2)||x_i-Dalpha_i||_2^2
        # D has shape (n_dimensions, n_components)
        # Alpha has shape (n_components, n_samples)
        err_matrix = reshaped_stacked_frames - D @ Alpha
        # each column i is the elementwise difference between the sample i
        # and its approximation by linear combination of D with weights Alpha_i
        mse = np.einsum('ij,ij->j', err_matrix, err_matrix).mean() # equivalent to np.mean(np.linalg.norm(err_matrix, axis=0, ord=2)**2)
        # print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Assignments computed, MSE={mse}")
        per_seed_results.append((mse, D, Alpha, init))
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Init {init+1}/{n_inits} done, MSE={mse:.4f}")
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Dictionary Learning fitting done")
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Starting stability analysis")

    mses, Ds, Alphas, inits = zip(*per_seed_results)

    (atoms_stability_matrix,
     atoms_order_stability_matrix,
     assignments_stability_matrix) = utils.compute_dictionary_stability(Ds, Alphas)

    utils.plot_dictionary_stability(
        mses=mses,
        atoms_stability_matrix=atoms_stability_matrix,
        atoms_order_stability_matrix=atoms_order_stability_matrix,
        assignments_stability_matrix=assignments_stability_matrix,
        n_subjects=n_subjs,
        savedir=savedir
    )

    utils.compare_assignments(
        assignments_stability_matrix,
        Alphas,
        mses,
        savedir=savedir
    )
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Stability analysis done")

    best_init_idx = np.argmin(mses)
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Best MSE={mses[best_init_idx]:.4f}")
    Alpha = Alphas[best_init_idx]
    D = Ds[best_init_idx]

    assignments = Alpha.T.toarray()  # (n_samples, k)
    comps = np.array(D.T, copy=False)


    del reshaped_stacked_frames
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
        optional_path_prefix='',
        load_retained_frames_df=False,
        n_comps=4,
        positive_code=False,
        alpha=2.0,
        subject_loading_n_workers=os.cpu_count() // 4,
        n_iters=300,
        positive_atoms=True,
        n_inits=50
    )

    main(
        group_path=paths.sample_derivatives,
        t=15,
        sel_mode='pos',
        optional_path_prefix='',
        load_retained_frames_df=False,
        n_comps=4,
        positive_code=False,
        alpha=2.0,
        subject_loading_n_workers=os.cpu_count() // 4,
        n_iters=300,
        positive_atoms=False,
        n_inits=50
    )

    main(
        group_path=paths.derivatives,
        t=15,
        sel_mode='pos',
        optional_path_prefix='',
        load_retained_frames_df=False,
        n_comps=4,
        positive_code=False,
        alpha=2.0,
        subject_loading_n_workers=os.cpu_count() // 4,
        n_iters=300,
        positive_atoms=False,
        n_inits=50
    )

    main(
        group_path=paths.derivatives,
        t=15,
        sel_mode='pos',
        optional_path_prefix='',
        load_retained_frames_df=False,
        n_comps=4,
        positive_code=False,
        alpha=2.0,
        subject_loading_n_workers=os.cpu_count() // 4,
        n_iters=300,
        positive_atoms=True,
        n_inits=50
    )

