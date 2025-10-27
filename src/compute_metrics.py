from datetime import datetime

from scipy.cluster.hierarchy import linkage, leaves_list

import paths, metrics
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import itertools

def main(
        expfolder,
        tr,
        plot_graphs
):
    frame_clustering = pd.read_pickle(expfolder / paths.frames_clustering_df_name)

    # increments clusters names so that 0 is free for baseline state
    frame_clustering['cluster_assignment'] = frame_clustering["cluster"] + 1

    K = int(frame_clustering['cluster_assignment'].max()); print(f"{K=}")

    n_states = K + 1 ; print(f"{n_states=}") # including baseline state

    max_frame_time = int(frame_clustering.index.get_level_values('frame_time').max()) + 1; print(f"{max_frame_time=}")

    if 'ses_name' not in frame_clustering.index.names:
        subjects = frame_clustering.index.get_level_values('subj_name').unique()
        full_idx = pd.MultiIndex.from_product([subjects, range(max_frame_time)],names=['subj','time'])

    else:
        pairs = frame_clustering.index.droplevel('frame_time').unique()
        times = range(max_frame_time)
        full_idx = pd.MultiIndex.from_tuples(
            [(subj, ses, t) for (subj, ses), t in itertools.product(pairs, times)],
            names=['subj_name', 'ses_name', 'time']
        )


    ssm = (frame_clustering['cluster_assignment']
           .reindex(full_idx)
           .fillna(0) # baseline is when the subject has no row at time
           .astype(np.int8)
           .unstack('time')
           )
    print(f"{ssm.shape=}")

    # test retained frames times are equal
    tm_retained_frames_indices = ssm.apply(lambda row: row[row != 0].index, axis=1)
    original_df_frame_indicies = pd.Series(frame_clustering.index.get_level_values('frame_time').groupby(frame_clustering.index.get_level_values('subj_name'))).rename_axis(index='subj')
    pd.testing.assert_series_equal(tm_retained_frames_indices, original_df_frame_indicies)
    # test cluster assignments are equals
    a = ssm.apply(lambda row: pd.Series(row[row != 0].values), axis=1).astype(int)
    b = frame_clustering.groupby(level="subj_name")['cluster_assignment'].apply(lambda x: pd.Series(x.values)).unstack().astype(int)
    b.index.name = 'subj'
    pd.testing.assert_frame_equal(a, b)
    # # test retained frames times are equal
    # tm_retained_frames_indices = ssm.apply(lambda row: row[row != 0].index, axis=1)
    # original_df_frame_indicies = pd.Series(frame_clustering.index.get_level_values('frame_time').groupby(frame_clustering.index.get_level_values('subj_name'))).rename_axis(index='subj')
    # pd.testing.assert_series_equal(tm_retained_frames_indices, original_df_frame_indicies)
    # # test cluster assignments are equals
    # a = ssm.apply(lambda row: pd.Series(row[row != 0].values), axis=1).astype(int)
    # b = frame_clustering.groupby(level="subj_name")['cluster_assignment'].apply(lambda x: pd.Series(x.values)).unstack().astype(int)
    # b.index.name = 'subj'
    # pd.testing.assert_frame_equal(a, b)


    results= []
    for subj, subj_state_seq in ssm.iterrows():

        switch_state_inds = np.where(
            subj_state_seq.values[:-1] != subj_state_seq.values[1:]
        )[0]

        seq_lengths = np.diff(switch_state_inds)*tr

        seq_type = subj_state_seq[switch_state_inds[:-1]].values

        n_seq_per_state = {state: count for state,count in  enumerate(np.bincount(seq_type, minlength=n_states))}

        seq_lengths_per_state = {
            state: seq_lengths[seq_type == state]
            for
            state in range(n_states)
        }

        avg_state_len = {
            state: seq_lengths_per_state[state].mean() if not seq_lengths_per_state[state].size == 0 else 0
            for
            state in range(n_states)
        }

        from_state = subj_state_seq[:-1].values
        to_state = subj_state_seq[1:].values
        trans_ids = from_state * n_states + to_state


        trans_counts = np.bincount(trans_ids, minlength=n_states * n_states)
        TPM_s = trans_counts.reshape(n_states, n_states)

        # normalize by all the transitions (max_frame_time - 1)
        TPM_s = TPM_s / (max_frame_time - 1)

        # TPM_s = TPM_s / TPM_s.sum(axis=1, keepdims=True)
        # TPM_s[np.isnan(TPM_s)] = 0

        results.append({
            'subj_name': subj,
            'n_seq_per_state': n_seq_per_state,
            'seq_lengths_per_state': seq_lengths_per_state,
            'avg_state_len': avg_state_len,
            'TPM': TPM_s
        })

    res_df = pd.DataFrame(results).set_index('subj_name')

    for col in [
        "CAPEntriesFromBaseline",
        "CAPExitsToBaseline",
        "CAPResilience" ,
        "BetweennessCentrality",
        "CAPInDegree",
        "CAPOutDegree",
        "frameCountsPerCAP",
        "frameFracPerCAP",
        "nxGraph"
    ]:
        res_df[col] = pd.Series(index=res_df.index, dtype=object)

    graph_dir = None
    if plot_graphs:
        graph_dir = expfolder / "graphs"
        graph_dir.mkdir(exist_ok=True)
        print(f"Graphs will be saved to \n{graph_dir}")

    for subj, tpm_s in res_df['TPM'].items():
        res_df.at[subj, "CAPEntriesFromBaseline"] = (
            metrics.CAPEntriesFromBaseline(tpm_s)
        )
        res_df.at[subj, "CAPExitsToBaseline"] = (
            metrics.CAPExitsToBaseline(tpm_s)
        )
        res_df.at[subj, "CAPResilience"] = (
            metrics.CAPResilience(tpm_s)
        )
        graph_plot_savepath = None
        if plot_graphs:
            graph_plot_savepath = graph_dir / f"{subj}.png"

        bc, g = metrics.BetweennessCentrality(tpm_s, graph_plot_savepath=graph_plot_savepath)

        res_df.at[subj, "BetweennessCentrality"] = (
            bc
        )

        res_df.at[subj, "nxGraph"] = g

        res_df.at[subj, "CAPInDegree"] = (
            metrics.CAPInDegree(tpm_s)
        )
        res_df.at[subj, "CAPOutDegree"] = (
            metrics.CAPOutDegree(tpm_s)
        )
        subj_state_sequence = ssm.loc[subj]
        counts = subj_state_sequence.value_counts()  # exclude baseline
        res_df.at[subj, "frameCountsPerCAP"] = counts.to_dict()
        res_df.at[subj, "frameFracPerCAP"] = (counts / counts.sum()).to_dict()

    per_cap_cols = [
        "CAPEntriesFromBaseline",
        "CAPExitsToBaseline",
        "CAPResilience" ,
        "BetweennessCentrality",
        "CAPInDegree",
        "CAPOutDegree",
        "frameCountsPerCAP",
        "frameFracPerCAP",
        "n_seq_per_state",
        "seq_lengths_per_state",
        "avg_state_len",
    ]
    global_cols = [
        "TPM",
        "nxGraph"
    ]

    assert set(res_df.columns) == set(per_cap_cols).union(set(global_cols))


    expanded = []
    for c in per_cap_cols:
        tmp = pd.json_normalize(res_df[c])
        tmp.columns = pd.MultiIndex.from_product([[c], tmp.columns])
        tmp.index = res_df.index
        expanded.append(tmp)

    out = pd.concat(expanded, axis=1)

    others = res_df.drop(columns=per_cap_cols)
    if len(others.columns):
        others.columns = pd.MultiIndex.from_tuples([(col, 0) for col in others.columns])
        out = pd.concat([others, out], axis=1)

    savepath = expfolder / paths.metrics_per_subject_df_name
    pd.to_pickle(out, savepath)
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] saved metrics df to \n{savepath}")
    return out

def plot_metrics(df):

    n_states = len(df["frameCountsPerCAP"].columns)

    without_baseline = False
    for col_name, n_subcols in df.columns.get_level_values(0).value_counts().items():
        if n_subcols == n_states:
            if col_name == "seq_lengths_per_state": continue
            fig, ax = plt.subplots(figsize=(8, 4))

            if without_baseline:
                data = df[col_name].iloc[:, 1:].to_numpy().astype(float)
                extent = (0.5, data.shape[1] + 0.5, 0, data.shape[0])
            else:
                data = df[col_name].to_numpy().astype(float)
                extent = (-0.5, data.shape[1] - 0.5, 0, data.shape[0])

            im = ax.imshow(data, aspect='auto', cmap='viridis', extent=extent)
            ax.set_title(col_name)
            ax.set_ylabel('Subjects')
            ax.set_xlabel('CAP State')
            fig.colorbar(im, ax=ax)
            ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
            plt.show()

    without_baseline = True
    for col_name, n_subcols in df.columns.get_level_values(0).value_counts().items():
        if n_subcols == n_states:
            if col_name == "seq_lengths_per_state": continue
            if without_baseline:
                avg_data = df[col_name].iloc[:, 1:].mean(axis=0)
                plot_range = range(1, n_states)
            else:
                avg_data = df[col_name].mean(axis=0)
                plot_range = range(0, n_states)

            fig, ax = plt.subplots(figsize=(6, 4))
            ax.bar(plot_range, avg_data)
            ax.set_title(col_name)
            ax.set_ylabel('frequency')
            ax.set_xlabel('CAP State')
            ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
            plt.show()



if __name__ == "__main__":
    expfolder = paths.sample_derivatives / "cust_kmeans_dist-correlation_ttype-percentage_tvalue-15_k-4_ninits-50_activation-pos_n-481"
    tr = 0.392
    df = main(expfolder, tr, plot_graphs=False)