from pathlib import Path
import paths


def main():
    non_preterm_path = paths.derivatives / "non_preterm"
    vol_dirs = sorted(non_preterm_path.glob("sub-*/vols/"))

    n_vol_dirs = len(vol_dirs)

    size_one_vol_dir = sum(f.stat().st_size for f in vol_dirs[0].rglob("*") if f.is_file())

    print(f"estimated size of all vol dirs: {n_vol_dirs * size_one_vol_dir / (1024 ** 3):.2f} GB")

    fullsize = 0
    for vol_dir in vol_dirs:
        total_size = sum(f.stat().st_size for f in vol_dir.rglob("*") if f.is_file())
        fullsize += total_size
        print(f"{vol_dir}: {total_size / (1024 ** 3):.2f} GB")
    print(f"Total size: {fullsize / (1024 ** 3):.2f} GB")

if __name__ == "__main__":
    main()