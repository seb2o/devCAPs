import re
import paths
import warnings


import os
from pathlib import Path
import nibabel as nib

def split_4d_to_3d(in_file: Path, out_dir: Path):
    """
    Split a 4D NIfTI file into multiple 3D volumes.

    Parameters
    ----------
    in_file : Path
        Path to the input 4D NIfTI file (.nii or .nii.gz).
    out_dir : Path
        Directory where the extracted 3D volumes will be written.

    Raises
    ------
    RuntimeError
        If the input file is not a 4D NIfTI image.

    Notes
    -----
    - Uses nibabel to load and process the input NIfTI file.
    - Output files are named as:
        <original_filename>_3D_<i>.nii
    - Volumes are indexed starting at 1.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    img4d = nib.load(str(in_file))
    if len(img4d.shape) != 4:
        raise RuntimeError(f"{in_file} is not 4D")

    stem = Path(in_file.stem).stem

    files_3d_already_here = sorted(out_dir.glob(f"{stem}_bold_3D_*.nii"))
    n_wrote = n_skipped = 0

    if len(files_3d_already_here) == 0:
        vols = nib.funcs.four_to_three(img4d)
        for i, vol in enumerate(vols, start=1):
            out_path = out_dir / f"{stem}_bold_3D_{i}.nii"
            vol.set_data_dtype(img4d.get_data_dtype())
            nib.save(vol, str(out_path))
            n_wrote += 1
    else:
        dataobj = img4d.dataobj  # array proxy -> loads only the needed 3D chunk
        n_vols = img4d.shape[3]

        # Reuse header + dtype; adjust shape per volume
        hdr_template = img4d.header.copy()
        hdr_template.set_data_dtype(img4d.get_data_dtype())

        for i in range(n_vols):
            out_path = out_dir / f"{stem}_bold_3D_{i + 1}.nii"
            if out_path.exists():
                n_skipped += 1
                continue

            vol_data = dataobj[..., i]  # lazy slice; no 4D load
            hdr = hdr_template.copy()
            hdr.set_data_shape(vol_data.shape)  # ensure 3D header
            vol_img = nib.Nifti1Image(vol_data, img4d.affine, header=hdr)
            vol_img.set_data_dtype(img4d.get_data_dtype())

            nib.save(vol_img, str(out_path))
            n_wrote += 1

    print(f"Wrote {n_wrote} volumes into {out_dir} | skipped {n_skipped} existing")

def main():
    """
    Main driver for splitting 4D BOLD fMRI files into 3D volumes.

    Workflow
    --------
    1. Resolves the derivatives directory defined in `DERIVATIVES_ROOT`.
    2. Iterates over all subject directories (those starting with `sub-`).
    3. Searches each subject for functional NIfTI files under `ses-*/func/*.nii.gz`.
    4. If multiple files are found, emits a warning and uses the first one.
    5. Calls `split_4d_to_3d` to split the chosen 4D file into 3D volumes.
    6. Stores the resulting volumes in a `vols/` subdirectory of the subject folder.

    Warnings
    --------
    - If no BOLD file is found for a subject, a warning is printed.
    - If multiple BOLD files are found, only the first one is used.
    """
    subj_dir = Path("/home/boo/capslifespan/data/derivatives/non_preterm").resolve()

    for subj in sorted(subj_dir.iterdir()):
        if not subj.is_dir() or not subj.name.startswith("sub-"):
            continue
        bolds = list(subj.glob("ses-*/func/*.nii.gz"))
        if not bolds:
            print(f"[WARN] No ses-*/func/*.nii.gz in {subj}")
            continue
        if len(bolds) > 1:
            warnings.warn(f"{subj} has more than one ses-*/func/*.nii.gz, using the first")

        bold = bolds[0]

        out_dir = subj / "vols"
        print(f"Processing {bold} -> {out_dir}")
        split_4d_to_3d(bold, out_dir)

if __name__ == "__main__":
    main()