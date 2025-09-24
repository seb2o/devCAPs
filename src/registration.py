import dataHandlers

def main():
    subs_to_keep = [
        "sub-CC00119XX12",
        "sub-CC00330XX09",
        "sub-CC00702AN09",
        "sub-CC00980XX16",
        "sub-CC00134XX11",
        "sub-CC00340XX11",
        "sub-CC00702BN09",
        "sub-CC00987XX23",
        "sub-CC00168XX12",
        "sub-CC00342XX13",
        "sub-CC00731XX14",
        "sub-CC01023XX09",
    ]
    template = "extdhcp40wk_lowres"
    datadir = "/home/menbuas/dvlp_cap/data/"
    derivativedir = "/home/menbuas/test_dir/"
    templates_paths={
        "extdhcp40wk_lowres":"/home/menbuas/test_warps_storage/extdhcp40wk_lowres.nii.gz"
    }

    dataset = dataHandlers.Dhcp3Fmri(
        root=datadir,
        derivative_root=derivativedir,
        templates_paths=templates_paths,
    )

    datastructure = dataset.get_raw_bolds_and_transforms_paths(
        template_name=template,
        subject_filter=lambda subject: subject.name in subs_to_keep,
        session_filter=lambda session: True,
    )
    dataset.pretty_print_dataset(datastructure)

if __name__ == "__main__":
    main()