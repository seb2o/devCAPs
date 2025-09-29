from pathlib import Path

atlas_url = "https://git.fmrib.ox.ac.uk/seanf/dhcp-resources/-/blob/master/docs/dhcp-augmented-volumetric-atlas-extended.md"


### REQUIRED STRUCTURE
# project_root/
# ├── data/
# │   ├── templates/
# │   │   ├── extdhcp40wk.nii.gz
# │   │   ├── extdhcp40wk_lowres.nii.gz
# │   │   ├── extdhcp40wk_parcellation.nii.gz
# │   │   └── extdhcp40wk_mask.nii.gz
# │   ├── derivatives/
# │   ├── preprocessed_bolds/
# │   └── bids_root/
# └── src/
#     └── paths.py  <- this file




src = Path(__file__).resolve().parent
absolute_root = src.parent
data = absolute_root / "data"

templates = data / "templates"
derivatives = data / "derivatives"
preprocessed_bolds = data / "preprocessed_bolds"
bids_root = data / "bids_root"

ext40Template = templates / "extdhcp40wk.nii.gz"
ext40Template_lowres = templates / "extdhcp40wk_lowres.nii.gz"
ext40Parcellation = templates / "extdhcp40wk_parcellation.nii.gz"
ext40Mask = templates / "extdhcp40wk_mask.nii.gz"


def rel(path: Path) -> str:
    """
    given an absolute path, returns the relative path starting from the root of the project
    :param path: the path that we want to convert to relative
    :return:
    """
    return str(path.relative_to(absolute_root))
