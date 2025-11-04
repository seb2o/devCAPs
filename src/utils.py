import csv
from datetime import datetime, timedelta
import os
import re
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from time import perf_counter
from typing import Callable
import nibabel as nib
import numpy as np
import pandas as pd
import paths

def get_bold_from_ses(ses_dir):
    bold_runs = [p for p in ses_dir.iterdir() if p.name.endswith('bold.nii.gz')]

    if len(bold_runs) == 0:
        raise FileNotFoundError(f"No bold.nii.gz files found in {ses_dir}")
    if len(bold_runs) > 1:
        warnings.warn(f"Found multiple bold.nii.gz files in {ses_dir}, using the first one: {bold_runs[0]}",
                      RuntimeWarning)
    return bold_runs[0]


def get_subj_and_ses_names_from_bolds(
        dir_path: Path,
        subject_filter = None
):
    """

    :param dir_path: path to the folder containing the bolds files named according to BIDS convention, w/o sub and ses subfolder structure
    :param subject_filter: function that takes the name of a subject, returns True if the subject should be included, False otherwise
    :return: dict with subjects names as keys and list of tuples (ses_name: [bold_path]) as values
    """

    if subject_filter is None:
        subject_filter = lambda x: True

    res = {}
    # each file is a potential bold file
    for file in dir_path.iterdir():
        if file.name.endswith('.nii.gz'):
            parts = file.stem.split('_')
            subj_name = Path(parts[0])
            ses_name = Path(parts[1])

            if not subject_filter(subj_name.name):
                continue

            if subj_name in res:
                res[subj_name].append((ses_name , file))
            else:
                res[subj_name] = [(ses_name , file)]
    return res

def get_ses_transform(bids_root, sub_name, ses_name, transform_pattern, transform_extension):
    transform_path = bids_root / sub_name / ses_name / 'xfm'

    if not bids_root.exists(): raise FileNotFoundError(f"BIDS root directory {bids_root} does not exist.")
    if not (bids_root / sub_name).exists(): raise FileNotFoundError(f"Subject directory {bids_root / sub_name} does not exist.")
    if not (bids_root / sub_name / ses_name).exists(): raise FileNotFoundError(f"Session directory {bids_root / sub_name / ses_name} does not exist")
    if not (bids_root / sub_name / ses_name / 'xfm').exists(): raise FileNotFoundError(f"Transform directory {bids_root / sub_name / ses_name / 'xfm'} does not exist.")

    transform_list = [
        p
        for p in transform_path.iterdir()
        if
        p.is_file()
        and transform_pattern in p.name
        and p.name.endswith(transform_extension)
    ]
    if len(transform_list) == 0:
        raise FileNotFoundError(f"No transform files found in {transform_path} matching pattern {transform_pattern} and extension {transform_extension}.")
    if len(transform_list) > 1:
        raise FileNotFoundError(f"Multiple transform files found in {transform_path} matching pattern {transform_pattern} and extension {transform_extension}.")
    return transform_list[0]

def build_dataset_info(
        bids_root,
        bolds_folder,
        template_name,
        transform_pattern,
        transform_extension,
        subject_filter=None,
):
    """

    :param bids_root: path to the root of a BUDS dataset containing the raw data and the xfm folders
    :param bolds_folder: path to the root of a folder containing bolds files named according to BIDS convention, wo sub and ses subfolder structure
    :param template_name: the template name as to how it is defined in the folder templates dict
    :param transform_pattern: the pattern that is used to identify the correct transform file in the xfm folder
    :param transform_extension: extension that the transform file should have, typically .mat or .nii.gz
    :param subject_filter: function that takes a subject name and returns True if the subject should be included, False otherwise    :return:
    """

    result = {}
    bids_root = Path(bids_root)
    bolds_folder = Path(bolds_folder)

    subj_ses_dict = get_subj_and_ses_names_from_bolds(
        bolds_folder,
        subject_filter
    )

    for subj_name, ses_bold_tuple_list in subj_ses_dict.items():
        res_subj_dict = {}
        for ses_name, bold_path in ses_bold_tuple_list:

            try:
                transform_path = get_ses_transform(
                    bids_root,
                    subj_name,
                    ses_name,
                    transform_pattern,
                    transform_extension
                )
                res_subj_dict[ses_name] = {
                    "bolds": [bold_path],
                    "transform": transform_path
                }
            except FileNotFoundError as e:
                print(f"{e} No transform file found for subject {subj_name}, session {ses_name}. Skipping this session.")
                continue

        if len(res_subj_dict) > 0:
            result[subj_name] = res_subj_dict
        else:
            print(f"No valid sessions found for subject {subj_name}. Skipping this subject.")

    dataset_infos = {
        "template_name": template_name,
        "tree": result
    }
    return dataset_infos

def default_session_tsv_filter(subject_dict: dict) -> bool:
    """
    Example filter function that checks if the 'age' field in the subject dictionary is >= 18.
    Adjust the logic as needed for your specific filtering criteria.
    """
    try:
        scan_age = float(subject_dict["scan_age"])
        radiology_score = float(subject_dict["radiology_score"])
    except (KeyError, ValueError) as e:
        warnings.warn(f"TSV filter error: {e}", RuntimeWarning)
        return False

    return scan_age >= 34.5 and radiology_score < 3.0


def filter_subject_with_session_tsv(
    subject: str,
    subjects_dir: str,
    tsv_filter: Callable[[dict], bool],
) -> bool:
    """
    Find the subject's folder (name contains `subject`), locate `<subject>_sessions.tsv`,
    and apply `filter_func` to it. Returns False if the TSV is missing.

    Raises:
        FileNotFoundError: if no matching subject folder is found.
    """

    subjects_dir = Path(subjects_dir)

    folder = next(
        (p for p in subjects_dir.iterdir() if p.is_dir() and subject in p.name),
        None,
    )
    if folder is None:
        raise FileNotFoundError(f"No folder found for subject {subject} in {subjects_dir}")

    tsv_file = folder / f"{subject}_sessions.tsv"
    if not tsv_file.exists():
        return False

    with open(tsv_file, newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        row = next(reader, None)

    if row is None:
        warnings.warn(f"{tsv_file} excluded: empty TSV file", RuntimeWarning)
        return False

    return tsv_filter(row)


def filter_subject_with_combined_tsv(subject, subjects_dir):
    """
    Filter subjects based on criteria from the combined.tsv file.
    Criteria: scan_age >= 34.5, birth_age >= 34.5, radiology_score < 3
    If multiple entries exist for a subject, the first one is used.
    :param subject:
    :param subjects_dir:
    :return:
    """
    subjects_dir = Path(subjects_dir)
    combined_tsv = subjects_dir / "combined.tsv"
    if not combined_tsv.exists():
        raise FileNotFoundError(f"{combined_tsv} does not exist.")
    with open(combined_tsv, newline='') as tsvfile:
        reader = csv.DictReader(tsvfile, delimiter='\t')
        subject_rows = [
            row for row in reader
            if 'sub-' + row['participant_id'] == subject
        ]
    if not subject_rows:
        warnings.warn(f"No entry found for subject {subject} in {combined_tsv}.")
        return False

    if len(subject_rows) > 1:
        warnings.warn(f"Multiple entries found for subject {subject} in {combined_tsv}. Using the first one.")

    row = subject_rows[0]
    try:
        scan_age = float(row["scan_age"])
        birth_age = float(row["birth_age"])
        radiology_score = float(row["radiology_score"])
    except (KeyError, ValueError) as e:
        warnings.warn(f"TSV filter error: {e}", RuntimeWarning)
        return False

    return (scan_age >= 34.5) and (birth_age >= 34.5) and (radiology_score < 3)


def get_masked_frames(vol_dir, gm_mask):
    """
    given vols dir path, assuming vols are suffixed with _3D_<frame_time>.nii
    applies the gm_mask to each volume and returns a list with frame_time as index
    raise ValueError if the mask shape does not match the volume shape
    :param vol_dir:
    :param gm_mask:
    :return:
    """
    pattern = re.compile(r'^.*_3D_(\d+)\.nii$')
    masked_vols = {}
    gm_mask_data = gm_mask.get_fdata().astype(bool)
    for vol_path in vol_dir.iterdir():
        if match := pattern.match(vol_path.name):
            frame_time = int(match.group(1))

            vol_data = nib.load(vol_path).get_fdata()

            if gm_mask_data.shape != vol_data.shape:
                raise ValueError(f"Mask shape {gm_mask_data.shape} does not match volume shape {vol_data.shape} for file {vol_path}")

            masked_data = vol_data * gm_mask_data
            masked_data = masked_data.flatten()

            masked_vols[frame_time]= masked_data
    return [masked_vols[i] for i in sorted(masked_vols.keys())]

def get_masked_frames_4d(bold_path, gm_mask):
    """
    given vols dir path, assuming vols are suffixed with _3D_<frame_time>.nii
    applies the gm_mask to each volume and returns a list with frame_time as index
    raise ValueError if the mask shape does not match the volume shape
    :param bold_path:
    :param gm_mask:
    :return:
    """
    gm_mask_data = gm_mask.get_fdata().astype(bool)[..., np.newaxis]
    timeserie_data = nib.load(bold_path).get_fdata()

    if gm_mask_data.shape[:-1] != timeserie_data.shape[:-1]:
        raise ValueError(f"Mask shape {gm_mask_data.shape} does not match volume shape {timeserie_data.shape} for file {bold_path}")

    masked_timeserie = timeserie_data * gm_mask_data

    return masked_timeserie



def get_masked_frames_4d_only_gm(bold_path, gm_mask):
    """
    given vols dir path, assuming vols are suffixed with _3D_<frame_time>.nii
    applies the gm_mask to each volume and returns a list with frame_time as index
    raise ValueError if the mask shape does not match the volume shape
    :param bold_path:
    :param gm_mask:
    :return:
    """
    gm_mask_data = gm_mask.get_fdata().astype(bool)
    timeserie =  nib.load(bold_path)
    timeserie_data = timeserie.get_fdata()


    if gm_mask_data.shape != timeserie_data.shape[:-1]:
        raise ValueError(f"Mask shape {gm_mask_data.shape} does not match volume shape {timeserie_data.shape} for file {bold_path}")

    masked_timeserie = timeserie_data[gm_mask_data]

    return masked_timeserie







def get_seed_timecourse(flat_vols, seed_mask, zscore=True):
    """
    assumes flat_vols is a list with timepoint as idx and flattened volume as value
    computes the mean timecourse within the seed mask
    and optionally zscore the timecourse

    :param flat_vols:
    :param seed_mask:
    :param zscore:
    :return:
    """
    seed_timecourse = []
    for vol in flat_vols:
        seed_voxels = vol[seed_mask.get_fdata().flatten() > 0]
        seed_timecourse.append(np.mean(seed_voxels))
    seed_timecourse = np.array(seed_timecourse)
    if zscore:
        seed_timecourse = (seed_timecourse - np.mean(seed_timecourse)) / np.std(seed_timecourse)
    return seed_timecourse


def get_seed_timecourse_from4d(timeserie, seed_mask, zscore=True):
    """
    assumes flat_vols is a list with timepoint as idx and flattened volume as value
    computes the mean timecourse within the seed mask
    and optionally zscore the timecourse

    :param timeserie:
    :param seed_mask:
    :param zscore:
    :return:
    """
    seed_mask_data = seed_mask.get_fdata().astype(bool)[..., np.newaxis]

    if seed_mask_data.shape[:-1] != timeserie.shape[:-1]:
        raise ValueError(f"Mask shape {seed_mask_data.shape} does not match volume shape {timeserie.shape} for seed mask {seed_mask}")

    seed_masked_timeserie = timeserie * seed_mask_data
    seed_timecourse = seed_masked_timeserie.mean(axis=(0,1,2))

    if zscore:
        seed_timecourse = (seed_timecourse - np.mean(seed_timecourse)) / np.std(seed_timecourse)
    return seed_timecourse


def get_seed_timecourse_from4d_only_gm(timeserie, gm_mask, seed_mask, zscore=True):
    """
    timeseries is a flattend, gm only 2d array (nGmvoxels, time)
    gm and seed mask are 3d nifti images
    computes the mean timecourse within the seed mask
    and optionally zscore the timecourse
    """

    gm_mask_data = gm_mask.get_fdata().astype(bool).reshape(-1)

    seed_mask_data = seed_mask.get_fdata().astype(bool).reshape(-1)

    seed_mask_gm = seed_mask_data[gm_mask_data]


    if seed_mask_gm.shape != timeserie.shape[:-1]:
        raise ValueError(f"seed mask shape {seed_mask_gm.shape} does not match volume shape {timeserie.shape} for seed mask {seed_mask}")

    seed_masked_timeserie = timeserie[seed_mask_gm] # (voxels_in_seed, time)
    seed_timecourse = seed_masked_timeserie.mean(axis=0) # (time,)

    if zscore:
        seed_timecourse = (seed_timecourse - np.mean(seed_timecourse)) / np.std(seed_timecourse, ddof=1 )
    return seed_timecourse




def get_percentile_thresholds(seed_timecourse, T):
    """
    computes the T and 100-T percentile of the seed timecourse
    :param seed_timecourse:
    :param T:
    :return:
    """
    lower_threshold = np.percentile(seed_timecourse, T, method="higher")
    upper_threshold = np.percentile(seed_timecourse, 100 - T, method="higher")
    return lower_threshold, upper_threshold


def unflatten_to_3d(cap, gm_mask, sample_volume, zscore=True):
    if np.prod(gm_mask.shape) != np.prod(cap.shape):
        raise ValueError(f"Shape of cap {cap.shape} does not match shape of gm_mask {gm_mask.shape}")

    if zscore:
        cap = (cap - np.mean(cap)) / np.std(cap)

    cap = cap.reshape(sample_volume.shape)

    cap = cap * gm_mask.get_fdata().astype(bool)

    return nib.Nifti1Image(cap, sample_volume.affine, sample_volume.header)


def unflatten_to_3d_only_gm(cap, gm_mask, sample_volume, zscore=True):

    flat_mask = gm_mask.get_fdata().astype(bool).flatten()

    if cap.ndim != 1 or cap.shape[0] != np.sum(flat_mask):
        raise ValueError(
            f"Mismatch: cap has shape {cap.shape}, but gm_mask has {np.sum(flat_mask)} GM voxels."
        )

    if zscore:
        cap = (cap - np.mean(cap)) / np.std(cap, ddof=1)

    full_cap = np.zeros(flat_mask.size, dtype=np.float32)
    full_cap[flat_mask] = cap

    cap_3d = full_cap.reshape(sample_volume.shape)


    return nib.Nifti1Image(cap_3d, sample_volume.affine, sample_volume.header)

def format_sec_for_print(sec):
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = sec % 60
    return_string = f"{s:.2f}s"
    if m > 0:
        return_string = f"{m:02d}m " + return_string
    if h > 0:
        return_string = f"{h:02d}h " + return_string
    return return_string

def extract_subject_frames(
        bold_path,
        gm_mask,
        seed_mask,
        T,
        selmode,
):

    if selmode != 'pos' and  selmode != 'neg' and  selmode != 'both':
        raise NotImplementedError(f"Selection mode {selmode} not implemented, only 'pos', 'neg' or 'both' are supported.")

    subj_name = bold_path.parent.parent.parent.name
    ses_name = bold_path.parent.parent.name

    start = perf_counter()
    masked_timeserie = get_masked_frames_4d_only_gm(bold_path, gm_mask)
    end = perf_counter()

    seed_timecourse = get_seed_timecourse_from4d_only_gm(masked_timeserie, gm_mask, seed_mask, zscore=True)

    l, h = get_percentile_thresholds(seed_timecourse, T)

    local_retained = []
    for frame_time in range(masked_timeserie.shape[-1]):
        seed_activity = seed_timecourse[frame_time]
        activity_type = frame_sign = None
        if seed_activity < l:
            if selmode in ['neg', 'both']: activity_type, frame_sign = "low", -1
        elif seed_activity > h:
            if selmode in ['pos', 'both']: activity_type, frame_sign = "high", 1
        if activity_type:
            local_retained.append((
                subj_name,
                ses_name,
                frame_time,
                activity_type,
                frame_sign * masked_timeserie[..., frame_time].flatten()
            ))

    return {
        "subj_name": subj_name,
        "ses_name": ses_name,
        "load_time": end - start,
        "n_vols": masked_timeserie.shape[1],
        "n_voxels": masked_timeserie.shape[0],
        "retained": local_retained,
    }


def get_frames(
        subj_4dbolds_paths,
        gm_mask,
        seed_mask,
        t,
        sel_mode,
        savedir,
        num_workers=1,
):

    retained_frames = []
    times = []
    start_time = perf_counter()
    with ThreadPoolExecutor(max_workers=min(os.cpu_count() // 2, num_workers)) as ex:
        futures = [
            ex.submit(extract_subject_frames, bold_path, gm_mask, seed_mask, t, sel_mode)
            for bold_path in subj_4dbolds_paths
        ]

        for fut in as_completed(futures):
            res = fut.result()
            times.append(res["load_time"])
            retained_frames.extend(res["retained"])
            now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            print(
                f"[{now}] Processed {res['subj_name']}-{res['ses_name']}, {res['n_vols']} vols, each of {res['n_voxels']}, in {format_sec_for_print(res['load_time'])}, retained {len(retained_frames)} frames so far"
            )

    total_time = perf_counter() - start_time

    print(
        f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]"
        f" Average time to load and mask a subject: {format_sec_for_print(np.mean(times))}."
        f" Total time: {format_sec_for_print(total_time)}"
        f" to process {len(subj_4dbolds_paths)} subjects."
        f" Parallelization speedup: {np.sum(times)/total_time:.2f}x"
    )

    retained_frames_df = pd.DataFrame(
        retained_frames,
        columns=[
            "subj_name",
            "ses_name",
            "frame_time",
            "type",
            "frame"
        ]
    ).set_index(
        [
            "subj_name",
            "ses_name",
            "frame_time"
        ]
    )

    retained_frames_df.to_pickle(savedir / paths.retained_frames_wo_clusters_df_name)
    print(
        f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]"
        f" Saved retained_frames_df to {savedir / paths.retained_frames_wo_clusters_df_name}"
    )


