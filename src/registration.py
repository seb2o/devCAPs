import utils, dataHandlers, paths

def main():


    template = "extdhcp40wk-lowres"
    datadir = paths.bids_root
    derivative_root=paths.derivatives
    templates_paths = {
        "extdhcp40wk-lowres": paths.ext40Template_lowres
    }

    folder = dataHandlers.Dhcp3Fmri(
        root=datadir,
        derivative_root=derivative_root,
        templates_paths=templates_paths,
    )

    dataset_info = utils.build_dataset_info(
        bids_root=datadir,
        bolds_folder=datadir,
        template_name=template,
        transform_pattern="from-bold_to-extdhcp40wk",
        transform_extension=".nii.gz",
        subject_filter=lambda subject: True
)

    folder.pretty_print_dataset(dataset_info['tree'])
    folder.normalize(
            dataset_infos=dataset_info,
            verbose=2
     )
if __name__ == "__main__":
    main()
