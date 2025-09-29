from pathlib import Path

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

            if not subject_filter(subj_name):
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