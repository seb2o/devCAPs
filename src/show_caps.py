#%%
import re

import paths
from nilearn import plotting, image
from pathlib import Path
import math, matplotlib.pyplot as plt

def plot_caps(
        folder_path: Path,
        template_img_path=paths.ext40Template,
        save_path= None,
        fig_title: str = None,
):
    zcaps_paths = sorted(list(folder_path.glob("CAP_*_z.nii")))
    for p in zcaps_paths:
        print(p.name, end=" | ")
    n = len(zcaps_paths)
    if n == 0:
        print("\nNo CAP_*_z.nii files found in the folder.")
        return

    fig, axes = plt.subplots(n, 1, figsize=(12, 3 * n))
    if n == 1:
        axes = [axes]

    bgimg = image.load_img(template_img_path)

    for ax, p in zip(axes, zcaps_paths):

        plotting.plot_stat_map(
            image.load_img(p),
            title=p.stem,
            display_mode="ortho",
            cut_coords=None,
            colorbar=True,
            vmax=5,
            cmap='RdBu_r',
            axes=ax,
            bg_img=bgimg,
            black_bg=False,
        )

    if not fig_title:
        fig_title = f"CAPs in {folder_path.name} ({n} total)"

    print(fig_title)

    fig.suptitle(fig_title, fontsize=20, y=0.92)


    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=300)
        print(f"\nFigure saved to: {save_path}")
    else:
        plt.show()


    fig, axes = plt.subplots(nrows=n, ncols=1, figsize=(12, 9 * n))
    if n  == 1:
        axes = [axes]


    for i in range(1, n+1):
            plot_cap_detail(
                folder_path,
                i,
                savedir=None,
                fig=fig,
                ax=axes[i-1],
            )
    fig.suptitle(f"Detailed CAPs in {folder_path.name} ({n} total)", fontsize=20, y=0.92)
    if save_path:
        detailed_save_path = save_path.parent / (save_path.stem + "_detailed.png")
        fig.savefig(detailed_save_path, bbox_inches=None, dpi=300)
        print(f"\nDetailed figure saved to: {detailed_save_path}")
    else:
        plt.show()


def plot_cap_detail(
        folder_path,
        cap_index: int,
        savedir=None,
        ax=None,
        fig=None
):
    cap_name = f"CAP_{cap_index:02d}_z"
    zcap_paths = sorted(list(folder_path.glob("CAP_*_z.nii")))
    zcap_path = [p for p in zcap_paths if p.stem == cap_name][0]

    if savedir is not None:
        savedir = savedir / (cap_name + ".png")

    plotting.plot_stat_map(
        zcap_path,
        title=zcap_path.stem,
        display_mode="mosaic",
        cut_coords=None,
        colorbar=True,
        vmax=5,
        bg_img=paths.ext40Template,
        black_bg=False,
        cmap="RdBu_r",
        output_file=savedir,
        figure=fig,
        axes=ax,
    )
    if not savedir and not ax:
        plt.show()


if __name__ == "__main__":

    folder_name = "euclidean_CAPS_k-4_tPercentage-15_activation-pos_n-34"

    m = re.search(r'k-(\d+).*?t(Percentage|p|t|Threshold)-(\d+).*?activation-(pos|neg).*?n-(\d+)', folder_name, re.IGNORECASE)
    k, ttype, tvalue, atype, n = m.groups()

    atype = 'high' if atype.lower() == 'pos' else 'low'

    if ttype.lower() == 'p'or ttype.lower() == 'percentage':
        t = f"top {int(tvalue)}%"
    else:
        t = f" gaussian threshold {tvalue}"


    title = f"MATLAB CAPs Euclidean dist  (k={k}) from {atype} activation ({t}) frames, n={n}"

    p = paths.sample_derivatives / folder_name

    #p = paths.sample_derivatives / "combined_caps_t_15"
    #title = "CAPs overview (python)"


    plot_caps(
        p,
        fig_title=title,
        save_path=p/(p.name + ".png"),
    )
