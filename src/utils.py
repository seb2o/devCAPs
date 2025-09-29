import csv
import warnings
from pathlib import Path
from typing import Callable


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

