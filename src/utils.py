from pathlib import Path

def get_subj_and_ses_names_from_bolds(dir_path: Path):
    res = {}
    for file in dir_path.iterdir():
        if file.name.endswith('.nii.gz'):
            parts = file.stem.split('_')
            subj_id = Path(parts[0])
            ses_id = Path(parts[1])

            if subj_id in res:
                res[subj_id].append({ses_id : file})
            else:
                res[subj_id] = [{ses_id : file}]
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
        raise ValueError(f"Multiple transform files found in {transform_path} matching pattern {transform_pattern} and extension {transform_extension}.")
    return transform_list[0]

def build_dataset_info(bids_root, bolds_folder, template_name, transform_pattern, transform_extension ):

    result = {}
    bids_root = Path(bids_root)
    bolds_folder = Path(bolds_folder)

    subj_ses_dict = get_subj_and_ses_names_from_bolds(bolds_folder)

    for subj_name, ses_dict_list in subj_ses_dict.items():
        res_subj_dict = {}
        for ses_dict in ses_dict_list:
            ses_name = list(ses_dict.keys())[0]
            bold_path = ses_dict[ses_name]
            transform_path = get_ses_transform(bids_root, subj_name, ses_name, transform_pattern, transform_extension)
            res_subj_dict[ses_name] = {
                "bolds": [bold_path],
                "transform": transform_path
            }
        result[subj_name] = res_subj_dict
    dataset_infos = {
        "template_name": template_name,
        "tree": result
    }
    return dataset_infos