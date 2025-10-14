import paths
from nilearn import plotting, image

def show_seed():
    seed_path = paths.ext40PosteriorCingulateGyrusMask

    plotting.plot_stat_map(
        seed_path,
        title=seed_path.stem,
        display_mode="mosaic",
        cut_coords=None,
        colorbar=True,
        cmap='RdBu_r',
        vmax=5,
        bg_img=paths.ext40Template,
        black_bg=False,
        output_file="seed.png"
    )

if __name__ == "__main__":
    show_seed()
