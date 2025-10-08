#%%
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


def plot_cap_detail(
        folder_path,
        cap_index: int,
):
    cap_name = f"CAP_{cap_index:02d}_z.nii"
    zcap_paths = sorted(list(folder_path.glob("CAP_*_z.nii")))
    zcap_path = [p for p in zcap_paths if p.name == cap_name][0]

    plotting.plot_stat_map(
        zcap_path,
        title=zcap_path.stem,
        display_mode="mosaic",
        cut_coords=None,
        colorbar=True,
        vmax=5,
        bg_img=paths.ext40Template,
        black_bg=False,
    )
    plt.show()


if __name__ == "__main__":

    p = paths.sample_derivatives / "non_preterm_CAPS_k-5_tPercentage-15_activation-pos_n-3"

    plot_caps(
        p,
        fig_title="high activation (top 15%) CAPs k=5 n=34",
        save_path=p/p.name
    )