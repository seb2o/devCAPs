import utils, dataHandlers, paths

def main():


    template = "extdhcp40wk-lowres"
    bids_root = paths.bids_root
    preprocessed_bolds_path = paths.preprocessed_bolds
    derivative_root=paths.derivatives
    templates_paths = {
        "extdhcp40wk-lowres": paths.ext40Template_lowres
    }

    folder = dataHandlers.Dhcp3Fmri(
        root=bids_root,
        derivative_root=derivative_root,
        templates_paths=templates_paths,
    )

    dataset_info = utils.build_dataset_info(
        bids_root=bids_root,
        bolds_folder=preprocessed_bolds_path,
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
