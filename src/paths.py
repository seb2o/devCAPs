import os.path as op

atlas_url = "https://git.fmrib.ox.ac.uk/seanf/dhcp-resources/-/blob/master/docs/dhcp-augmented-volumetric-atlas-extended.md"

src = op.dirname(op.abspath(__file__))
absolute_root = op.dirname(src)
data = op.join(absolute_root, "data")

ext40Template = op.join(data, "extdhcp40wk.nii.gz")
ext40Template_lowres = op.join(data, "extdhcp40wk_lowres.nii.gz")
ext40Parcellation = op.join(data, "extdhcp40wk_parcellation.nii.gz")
ext40Mask = op.join(data, "extdhcp40wk_mask.nii.gz")


def rel(path):
    """
    given an absolute path, returns the relative path starting from the root of the project
    :param path: the path that we want to convert to relative
    :return:
    """
    return op.relpath(path, absolute_root)