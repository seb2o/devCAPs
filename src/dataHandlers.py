import os
import warnings
from datetime import timedelta
from typing import Optional
from pathlib import Path
import paths
from fsl.wrappers.fnirt import applywarp
import time

class Dhcp3Fmri:
    """
    Assumes BIDS format and dHCP rel 3 naming conventions.
    """

    def __init__(self,
                 root,
                 derivative_root=None,
                 templates_paths=None,
                 bold_file_suffix="desc-preproc_bold.nii.gz"
                 ):

        self.datasets_infos = {}

        if templates_paths is None:
            self.templates_paths = {
                "extdhcp40wk": paths.ext40Template,
                "extdhcp40wk_lowres": paths.ext40Template_lowres,
            }
        else:
            self.templates_paths = templates_paths

        for name, path in self.templates_paths.items():
            if not Path(path).is_file():
                raise ValueError(
                    f"Template {name} not found at {path}."
                    f"Download from {paths.atlas_url}."
                )

        self.root = Path(root)

        self.derivative_root = Path(derivative_root) if derivative_root is not None else self.root / "derivatives"

        self.bold_file_suffix = bold_file_suffix

    def get_template(self, template):
        if template not in self.templates_paths.keys():
            raise ValueError(f"Template {template} not found in {self.templates_paths}"
                             f". Available templates: {list(self.templates_paths.keys())}")
        return self.templates_paths[template]

    def get_subjects_paths(self, subject_filter, keep_non_folder=False):

        path_filter = lambda path: path.name.startswith("sub-") and subject_filter(path)
        if not keep_non_folder:
            base_filter = path_filter
            path_filter = lambda path: base_filter(path) and path.is_dir()

        return sorted([
            p
            for p in self.root.iterdir()
            if
            path_filter(p)
        ])

    @staticmethod
    def get_sessions_paths_from_subject(subj_path: Path, session_filter):

        return sorted([
            p
            for p in subj_path.iterdir()
            if
            p.is_dir()
            and p.name.startswith("ses-")
            and session_filter(p)
        ])

    @staticmethod
    def get_transform_path_from_session_path(ses_path: Path, template_name):
        xfm_dir = ses_path / "xfm"
        if not xfm_dir.is_dir():
            raise ValueError(f"xfm directory not found in {xfm_dir}")
        else:
            transform_list = [
                p
                for p in xfm_dir.iterdir()
                if
                p.is_file()
                and "from-bold_to-" in p.name
                and template_name.split('_')[0] in p.name
            ]
            if len(transform_list) == 0:
                raise ValueError(f"No transform found in {xfm_dir} for template {template_name}")
            elif len(transform_list) > 1:
                raise ValueError(
                    f"Multiple transforms found in {xfm_dir} for template {template_name}. Using the first one.")
            else:
                return transform_list[0]

    def pretty_print_dataset(self, dataset_dict=None, dataset_name=None):

        if dataset_dict is None:
            if dataset_name is None:
                if 'default' not in self.datasets_infos:
                    raise ValueError("No dataset_dict provided and no default dataset found."
                                     "Please run get_raw_bolds_and_transforms_paths first "
                                     "or provide a dataset_dict or valid dataset_name.")
                else:
                    dataset_dict = self.datasets_infos['default']['tree']

            elif dataset_name not in self.datasets_infos:
                raise ValueError(f"Dataset {dataset_name} not found in datasets_infos.")
            else:
                dataset_dict = self.datasets_infos[dataset_name]['tree']

        for subject_path, sessions_dict in dataset_dict.items():
            print(f"Subject: {subject_path.name}")
            for session_path, session_data in sessions_dict.items():
                print(f"  Session: {session_path.name}")
                bolds = [b.name for b in session_data['bolds']]
                normalized_bolds = None
                transform = session_data['transform'].name

                if 'normalized_bolds' in session_data:
                    normalized_bolds = [b.name for b in session_data['normalized_bolds']]

                print(f"    Transform: {transform}")
                for runid, run in enumerate(bolds):
                    print(f"    Run {runid}: {run}")
                    if normalized_bolds:
                        print(f"    Normalized: {normalized_bolds[runid]}")


    def get_bolds_paths_from_session_path(self, ses_path):
        func_dir = ses_path / "func"
        if not func_dir.is_dir(): raise ValueError(f"func directory not found in {func_dir}")
        return sorted([
            p
            for
            p in func_dir.iterdir()
            if
            p.is_file()
            and p.name.endswith(self.bold_file_suffix)
        ])

    def get_normalized_bold_path(self, subject, session, run, template_name):
        file_folder = self.derivative_root / subject.name / session.name / "func"
        os.makedirs(file_folder, exist_ok=True)
        file_path = file_folder / f"{subject.name}_{session.name}_task-rest_run-{run + 1}_space-{template_name}_bold.nii.gz"
        return file_path


    def get_raw_bolds_and_transforms_paths(self,
                                           template_name: str,
                                           subject_filter,
                                           session_filter,
                                           dataset_name = 'default'
                                           ):

        results = {}
        for subj_path in self.get_subjects_paths(subject_filter=subject_filter):
            sessions_paths = self.get_sessions_paths_from_subject(
                subj_path, session_filter=session_filter
            )
            if len(sessions_paths) == 0:
                warnings.warn(f"No sessions found for subject {subj_path.name} after applying the session filter.")
                continue

            per_subject = {}
            for ses_path in sessions_paths:

                bolds = self.get_bolds_paths_from_session_path(ses_path)
                if len(bolds) == 0: raise ValueError(f"No bolds found for session {ses_path.name} of subject {subj_path.name}.")

                transform_path = self.get_transform_path_from_session_path(ses_path, template_name)

                per_subject[ses_path] = {
                    "bolds": bolds,
                    "transform": transform_path
                }
            results[subj_path] = per_subject

        dataset_infos = {
            "template_name": template_name,
            "tree": results
        }

        self.datasets_infos[dataset_name] = dataset_infos
        return dataset_infos


    def normalize(self,
                  dataset_infos: Optional[dict] = None,
                  verbose=2
                  ):
        """
        Normalize all the bold runs for which a transform to the given template exists. Checks if the normalized bold already exists
        :return: a nested dictionary containing names of the subjects, sessions and runs that were normalized as keys and
        the path to the normalized bold as values.
        dictionary structure:
        {
            subject1: {
                session1: {
                    'bolds' : [run1, run2, ...],
                    'transform': transform_path
                },
                session2: {
                    'bolds' : [run1, run2, ...],
                    'transform': transform_path
                },
                ...
            },
            subject2: {
                session1: {
                    'bolds' : [run1, run2, ...],
                    'transform': transform_path
                },
                ...
            },
            ...
        }

        """

        if dataset_infos is None:
            if 'default' not in self.datasets_infos:
                raise ValueError("No dataset_tree_dict provided and no default dataset found."
                                 "Please run get_raw_bolds_and_transforms_paths first "
                                 "or provide a dataset_tree_dict.")

            dataset_infos = self.datasets_infos['default']

        dataset_tree_dict = dataset_infos['tree']
        template_name = dataset_infos['template_name']

        if verbose>1:
            self.pretty_print_dataset(dataset_tree_dict)

        already_normalized_count = 0
        normalized_count = 0



        for subject_path, sessions_dict in dataset_tree_dict.items():
            for session_path, session_data in sessions_dict.items():
                bolds = session_data['bolds']
                transform = session_data['transform']

                session_data['normalized_bolds'] = []

                for runid, run in enumerate(bolds):
                    # check if normalized bold already exists
                    outpath: Path = self.get_normalized_bold_path(subject_path, session_path, runid, template_name)
                    starttime = time.time()
                    if outpath.exists():
                        already_normalized_count += 1
                        if verbose>1:
                            print(f"{outpath} already exists, skipping.")

                    else:
                        # apply normalization using fsl.wrappers.fnirt.applywarp
                        applywarp(
                            src=run,
                            ref=self.get_template(template_name),
                            warp=transform,
                            out=outpath
                        )
                        normalized_count += 1
                        if verbose>1:
                            print(f"normalized {outpath}")
                    endtime = time.time()
                    if verbose>1:
                        duration = endtime - starttime
                        print(f"took {str(timedelta(seconds=int(duration)))} H:MM:SS")
                    session_data['normalized_bolds'].append(outpath)



        print(f"Normalized: {normalized_count} | Skipped (already exists): {already_normalized_count}")

        return dataset_infos


