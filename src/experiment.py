import paths, report_caps, get_caps_4d_loading


def main(
        group_path=paths.sample_derivatives,
        t=15,
        threshold_type='percentage',
        n_clusters=4,
        n_inits=50,
        sel_mode='pos',
        cluster_dist='euclidean',
        load_retained_frames_df=False,
        recompute_clusters=True
):
    exp_path = get_caps_4d_loading.main(
        group_path=group_path,
        t=t,
        threshold_type=threshold_type,
        n_clusters=n_clusters,
        n_inits=n_inits,
        sel_mode=sel_mode,
        cluster_dist=cluster_dist,
        load_retained_frames_df=load_retained_frames_df,
        recompute_clusters=recompute_clusters,
    )
    report_caps.main(exp_path)

if __name__ == "__main__":
    main()
