import paths, report_caps, get_caps_4d_loading, compute_metrics


def main(
        group_path,
        t,
        threshold_type,
        n_clusters,
        n_inits,
        sel_mode,
        cluster_dist,
        optional_path_prefix="",
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
        optional_path_prefix=optional_path_prefix,
        load_retained_frames_df=load_retained_frames_df,
        recompute_clusters=recompute_clusters,
    )

    compute_metrics.main(
        expfolder=exp_path,
        tr=0.392,
        plot_graphs=True
    )

    report_caps.main(exp_path)

if __name__ == "__main__":
    main(
        group_path=paths.sample_derivatives,
        t=15,
        threshold_type='percentage',
        n_clusters=4,
        n_inits=50,
        sel_mode='both',
        cluster_dist='euclidean',
        optional_path_prefix="sklearn_kmeans_"
    )
