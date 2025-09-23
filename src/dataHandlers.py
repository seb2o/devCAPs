import warnings
from typing import Optional
from pathlib import Path
from tqdm import tqdm
import paths
from fsl.wrappers.fnirt import applywarp


class Dhcp3Fmri:
    """
    Assumes BIDS format and dHCP rel 3 naming conventions.
    """

    def __init__(self, root, bold_file_suffix="desc-preproc_bold.nii.gz"):

        self.datasets = {}
        self.templates_paths = {
            "extdhcp40wk": paths.ext40Template,
            "extdhcp40wk_lowres": paths.ext40Template_lowres,
        }

        # check that all templates exist
        for name, path in self.templates_paths.items():
            if not Path(path).is_file():
                raise ValueError(
                    f"Template {name} not found at {path}."
                    f"Download from {paths.atlas_url} and place in {paths.data}."
                )

        self.root = Path(root)
        self.bold_file_suffix = bold_file_suffix

    def get_template(self, template):
        if template not in self.templates_paths.keys():
            raise ValueError(f"Template {template} not found in {self.templates_paths}"
                             f". Available templates: {list(self.templates_paths.keys())}")
        return self.templates_paths[template]

    def get_subjects_paths(self, subject_filter):
        return sorted([
            p
            for p in self.root.iterdir()
            if
            p.is_dir()
            and p.name.startswith("sub-")
            and subject_filter(p)
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
                and f"from-bold_to-{template_name}" in p.name
            ]
            if len(transform_list) == 0:
                raise ValueError(f"No transform found in {xfm_dir} for template {template_name}")
            elif len(transform_list) > 1:
                raise ValueError(
                    f"Multiple transforms found in {xfm_dir} for template {template_name}. Using the first one.")
            else:
                return transform_list[0]

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
        return self.root / "derivatives" / subject.name / session.name / "func" / f"{subject.name}_{session.name}_task-rest_run-{run + 1}_space-{template_name}_bold.nii.gz"

    def get_raw_bolds_and_transforms_paths(self,
                                           template_name: str,
                                           subject_filter,
                                           session_filter,
                                           dataset_name: Optional[str] = None
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
        self.datasets[dataset_name] = results
        return results


    def normalize(self,
                  template_name,
                  subject_filter,
                  session_filter,

                  ):
        """
        Normalize all the bold runs for which a transform to the given template exists. Checks if the normalized bold already exists
        :param template_name: the template name
        :param subject_filter: function to apply to the subject folder to determine if the subject is kept
        :param session_filter: function to apply to the session folder to determine if the session is kept
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

        dataset = self.get_raw_bolds_and_transforms_paths(
            template_name=template_name,
            subject_filter=subject_filter,
            session_filter=session_filter
        )

        already_normalized_count = 0
        normalized_count = 0
        normalized_paths = []

        for subject, sessions in dataset.items():
            for session in sessions:
                bolds = session['bolds']
                transform = session['transform']
                for runid, run in enumerate(bolds):
                    # check if normalized bold already exists
                    outpath: Path = self.get_normalized_bold_path(subject, session, runid, template_name)

                    if outpath.exists():
                        already_normalized_count += 1
                        continue
                    else:
                        # apply normalization using fsl.wrappers.fnirt.applywarp
                        applywarp(
                            in_file=run,
                            ref_file=self.get_template(template_name),
                            warp_file=transform,
                            out_file=outpath
                        )
                        normalized_count += 1

                    normalized_paths.append(outpath)
        print(f"Normalized: {normalized_count} | Skipped (already exists): {already_normalized_count}")
        return normalized_paths


