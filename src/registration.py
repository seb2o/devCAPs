import utils
import dataHandlers

def main():


    template = "extdhcp40wk-lowres"
    datadir = "/home/boo/kebiri/rel3_dhcp_fmri_pipeline/"
    derivativedir = "/home/boo/capslifespan/data/derivatives/"
    templates_paths={
        "extdhcp40wk-lowres":"/home/boo/capslifespan/data/templates/extdhcp40wk_lowres.nii.gz",
    }

    folder = dataHandlers.Dhcp3Fmri(
        root=datadir,
        derivative_root=derivativedir,
        templates_paths=templates_paths,
    )

    dataset_info = utils.build_dataset_info(
        bids_root=datadir,
        bolds_folder="/home/boo/kebiri/preproc_fMRI_dHCP/",
        template_name="extdhcp40wk-lowres",
        transform_pattern="from-bold_to-extdhcp40wk",
        transform_extension=".nii.gz",
        subject_filter=lambda subject: True
)

#    datastructure = dataset.get_raw_bolds_and_transforms_paths(
#        template_name=template,
#        subject_filter=lambda subject: subject.name in subs_to_keep,
#        session_filter=lambda session: True,
#    )

    folder.pretty_print_dataset(dataset_info['tree'])
    folder.normalize(
            dataset_infos=dataset_info,
            verbose=2
     )
if __name__ == "__main__":
    main()
