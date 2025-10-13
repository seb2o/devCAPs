from pathlib import Path
import paths


def main():
    non_preterm_path = paths.derivatives / "non_preterm"
    vol_dirs = sorted(non_preterm_path.glob("sub-*/vols/"))
    fullsize = 0
    for vol_dir in vol_dirs:
        total_size = sum(f.stat().st_size for f in vol_dir.rglob("*") if f.is_file())
        fullsize += total_size
        print(f"{vol_dir}: {total_size / (1024 ** 3):.2f} GB")
    print(f"Total size: {fullsize / (1024 ** 3):.2f} GB")

if __name__ == "__main__":
    main()