from __future__ import annotations

import csv
import re
import warnings
from pathlib import Path
from typing import Callable, List

import paths


def non_preterm_criterion(row):
    birth_age = float(row["birth_age"])

    return birth_age >= 37.5

def preterm_criterion(row):
    birth_age = float(row["birth_age"])

    return birth_age < 37.5

def get_subjs_to_symlink(
    bids_root: Path,
    derivatives_path: Path,
    group_tsv_criterion: Callable[[dict], bool],
    subject_filter: Callable[[str], bool] | None = None,
) -> List[Path]:
    """
    Return subject directories in `derivatives_path` that pass `group_tsv_criterion`
    based on rows in `bids_root/combined.tsv`, matched by (participant_id, session_id)
    inferred from the first (sorted) ses-*/func/*.nii.gz per subject.
    """
    combinedtsv_path = bids_root / "combined.tsv"
    if not combinedtsv_path.exists():
        raise FileNotFoundError(f"combined.tsv not found in {bids_root}.")

    with combinedtsv_path.open("r", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        rows = list(reader)

    required_cols = {"participant_id", "session_id"}
    missing = required_cols - set(rows[0].keys()) if rows else required_cols
    if missing:
        raise KeyError(f"combined.tsv missing required columns: {sorted(missing)}")

    index = {}
    for row in rows:
        pid = row["participant_id"]
        sid = row["session_id"]
        index.setdefault((pid, sid), []).append(row)

    subjs_to_symlink: List[Path] = []

    for subject_path in sorted(p for p in derivatives_path.iterdir() if p.is_dir()):
        name = subject_path.name
        if not name.startswith("sub-"):
            continue
        if subject_filter and not subject_filter(name):
            warnings.warn(f"Skipped {name} by subject filter.", category=UserWarning)
            continue

        bolds = sorted(subject_path.glob("ses-*/func/*.nii.gz"))
        if not bolds:
            warnings.warn(f"No ses-*/func/*.nii.gz in {subject_path}", category=UserWarning)
            continue
        if len(bolds) > 1:
            warnings.warn(f"{subject_path} has >1 func NIfTI; using the first after sort.", category=UserWarning)

        bold = bolds[0]
        bold_str = bold.name

        m_ses = re.search(r"ses-([^_/]+)", bold_str)
        m_sub = re.search(r"sub-([^_/]+)", bold_str)
        if not (m_sub and m_ses):
            warnings.warn(f"Could not parse sub-/ses- from {bold}", category=UserWarning)
            continue

        subj_id = m_sub.group(1)
        sess_id = m_ses.group(1)

        matches: list[dict] = index.get((subj_id, sess_id), [])
        if not matches:
            warnings.warn(f"No TSV entry for {subj_id} {sess_id} in {combinedtsv_path}.", category=UserWarning)
            continue
        if len(matches) > 1:
            warnings.warn(f"Multiple TSV entries for {subj_id} {sess_id}; using the first.", category=UserWarning)

        row = matches[0]
        try:
            if group_tsv_criterion(row):
                subjs_to_symlink.append(subject_path)
            else:
                print(f"Excluded {subj_id} by TSV criterion. (birth_age: {row['birth_age']})")
        except (KeyError, ValueError) as e:
            warnings.warn(f"TSV filter error for {subj_id}: {e}", category=RuntimeWarning)
            continue


    return subjs_to_symlink

def main(
        group_name,
        group_tsv_criterion,
        bids_root=paths.bids_root,
        derivatives_path=paths.derivatives,
        subject_filter=None,
):
    kept_subjs = get_subjs_to_symlink(
        bids_root,
        derivatives_path,
        group_tsv_criterion,
        subject_filter,
    )

    group_dir = derivatives_path / group_name
    group_dir.mkdir(exist_ok=True)

    for subj_path in kept_subjs:
        link_path = group_dir / subj_path.name
        if link_path.exists():
            print(f"Link {link_path} already exists, skipping.")
            continue
        print(f"Linking {subj_path} to {link_path}")
        link_path.symlink_to(subj_path, target_is_directory=True)






if __name__ == '__main__':
    main(
        group_name="preterm",
        group_tsv_criterion=preterm_criterion,
        bids_root=paths.bids_root,
        derivatives_path=paths.derivatives
)
