from pathlib import Path
import markdown, weasyprint
import paths
import nibabel as nib
import numpy as np
import pandas as pd



LUT = {
1 :"Hippocampus left",
2 :"Hippocampus right",
3 :"Amygdala left",
4 :"Amygdala right",
5 :"Anterior temporal lobe, medial part left GM",
6 :"Anterior temporal lobe, medial part right GM",
7 :"Anterior temporal lobe, lateral part left GM",
8 :"Anterior temporal lobe, lateral part right GM",
9 :"Gyri parahippocampalis et ambiens anterior part left GM",
10:"Gyri parahippocampalis et ambiens anterior part right GM",
11:"Superior temporal gyrus, middle part left GM",
12:"Superior temporal gyrus, middle part right GM",
13:"Medial and inferior temporal gyri anterior part left GM",
14:"Medial and inferior temporal gyri anterior part right GM",
15:"Lateral occipitotemporal gyrus, gyrus fusiformis anterior part left GM",
16:"Lateral occipitotemporal gyrus, gyrus fusiformis anterior part right GM",
17:"Cerebellum left",
18:"Cerebellum right",
19:"Brainstem, spans the midline",
20:"Insula right GM",
21:"Insula left GM",
22:"Occipital lobe right GM",
23:"Occipital lobe left GM",
24:"Gyri parahippocampalis et ambiens posterior part right GM",
25:"Gyri parahippocampalis et ambiens posterior part left GM",
26:"Lateral occipitotemporal gyrus, gyrus fusiformis posterior part right GM",
27:"Lateral occipitotemporal gyrus, gyrus fusiformis posterior part left GM",
28:"Medial and inferior temporal gyri posterior part right GM",
29:"Medial and inferior temporal gyri posterior part left GM",
30:"Superior temporal gyrus, posterior part right GM",
31:"Superior temporal gyrus, posterior part left GM",
32:"Cingulate gyrus, anterior part right GM",
33:"Cingulate gyrus, anterior part left GM",
34:"Cingulate gyrus, posterior part right GM",
35:"Cingulate gyrus, posterior part left GM",
36:"Frontal lobe right GM",
37:"Frontal lobe left GM",
38:"Parietal lobe right GM",
39:"Parietal lobe left GM",
40:"Caudate nucleus right",
41:"Caudate nucleus left",
42:"Thalamus right, high intensity part in T2",
43:"Thalamus left, high intensity part in T2",
44:"Subthalamic nucleus right",
45:"Subthalamic nucleus left ",
46:"Lentiform Nucleus right",
47:"Lentiform Nucleus left",
48:"Corpus Callosum",
49:"Lateral Ventricle left",
50:"Lateral Ventricle right",
51:"Anterior temporal lobe, medial part left WM",
52:"Anterior temporal lobe, medial part right WM",
53:"Anterior temporal lobe, lateral part left WM",
54:"Anterior temporal lobe, lateral part right WM",
55:"Gyri parahippocampalis et ambiens anterior part left WM",
56:"Gyri parahippocampalis et ambiens anterior part right WM",
57:"Superior temporal gyrus, middle part left WM",
58:"Superior temporal gyrus, middle part right WM",
59:"Medial and inferior temporal gyri anterior part left WM",
60:"Medial and inferior temporal gyri anterior part right WM",
61:"Lateral occipitotemporal gyrus, gyrus fusiformis anterior part left WM",
62:"Lateral occipitotemporal gyrus, gyrus fusiformis anterior part right WM",
63:"Insula right WM",
64:"Insula left WM",
65:"Occipital lobe right WM",
66:"Occipital lobe left WM",
67:"Gyri parahippocampalis et ambiens posterior part right WM",
68:"Gyri parahippocampalis et ambiens posterior part left WM",
69:"Lateral occipitotemporal gyrus, gyrus fusiformis posterior part right WM",
70:"Lateral occipitotemporal gyrus, gyrus fusiformis posterior part left WM",
71:"Medial and inferior temporal gyri posterior part right WM",
72:"Medial and inferior temporal gyri posterior part left WM",
73:"Superior temporal gyrus, posterior part right WM",
74:"Superior temporal gyrus, posterior part left WM",
75:"Cingulate gyrus, anterior part right WM",
76:"Cingulate gyrus, anterior part left WM",
77:"Cingulate gyrus, posterior part right WM",
78:"Cingulate gyrus, posterior part left WM",
79:"Frontal lobe right WM",
80:"Frontal lobe left WM",
81:"Parietal lobe right WM",
82:"Parietal lobe left WM",
83:"CSF",
84:"Extra-cranial background",
85:"Intra-cranial background",
86:"Thalamus right, low intensity part in T2",
87:"Thalamus left, low intensity part in T2",
}

def main(
        data_path
):

    seed_nodes = [
        node_name for node_name in LUT.values()
        if
            not "WM" in node_name
            and any(
                subn in node_name for subn in ["Cingulate gyrus, posterior"]
            )
    ]
    print(f"seed nodes: {seed_nodes}")


    parcell_path = paths.ext40Parcellation_lowres
    parcell = nib.load(parcell_path)
    parcell_data = parcell.get_fdata()

    gm_mask_path = paths.ext40GreyMatterMask
    gm_mask_data = nib.load(gm_mask_path).get_fdata().astype(bool)

    caps_img_paths = sorted(data_path.glob("CAP_*_z.png"))
    cap_img_paths = {p.stem.removesuffix("_z") :p for p in caps_img_paths}
    cap_paths = sorted(data_path.glob("CAP_*_z.nii"))
    n_caps = len(cap_paths)
    print(f"Found {n_caps} CAP files.")

    detailed_overview_paths = sorted(data_path.glob("*_detailed.png"))

    detailed_overview_path = detailed_overview_paths[0]
    print(f"Using detailed overview file: {detailed_overview_path.name}")

    cluster_sizes_path = data_path / "cluster_sizes.pkl"
    cluster_sizes = pd.read_pickle(cluster_sizes_path)
    cluster_sizes.index = [f"CAP_{i+1:02d}" for i in cluster_sizes.index]
    n_frames = cluster_sizes.sum()

    glob_res = {}
    for cap_idx, cap_path in enumerate(cap_paths):
        print(f"Processing {cap_idx+1}/{n_caps}: {cap_path.name}")
        cap = nib.load(cap_path)
        cap_data = cap.get_fdata()

        # check that GM mask has been properly applied
        if not np.all(cap_data[~gm_mask_data] == 0):
            raise ValueError(f"CAP file {cap_path.name} contains non-zero values outside GM mask.")

        cap_key = f"CAP_{cap_idx+1:02d}"
        glob_res[cap_key] = {}

        for node_idx, node_name in LUT.items():

            # only process GM
            # if "WM" in node_name:
            #     continue

            node_mask = parcell_data == node_idx
            if np.sum(node_mask) == 0:
                print(f"{cap_path.name} - {node_name}: No voxels found in parcellation.")
                continue
            if node_name in seed_nodes:
                mean_cap_value = 0
            else:
                node_cap_values = cap_data[node_mask]
                mean_cap_value = np.mean(node_cap_values)
            glob_res[cap_key][node_name] = mean_cap_value

    df = pd.DataFrame(glob_res)

    # Write top/bottom 10 per column to a Markdown file, embedding the corresponding image
    outfile = data_path /  "report.md"
    lines = []

    lines.append(f"# CAP Analysis Report\n")
    lines.append(f"Folder: `{data_path}`\n")

    for col in df.columns:
        lines.append(f"### {col}\n")

        # cluster size
        lines.append(f"**Cluster size:** {cluster_sizes.loc[col]} frames\n")

        # Image
        img_path = cap_img_paths[col].relative_to(data_path)
        lines.append(f"![{col}]({img_path.as_posix()})")
        lines.append("\n---\n")

        # Top 5
        lines.append("**Top 10 nodes**")
        lines.append("\n| Node | Value |")
        lines.append("|---|---:|")
        top5 = df[col].sort_values(ascending=False).head(10)
        for idx, val in top5.items():
            lines.append(f"| `{str(idx)}` | {val:.4f} |")
        lines.append("")

        # Bottom 5
        lines.append("**Bottom 10 nodes**")
        lines.append("\n| Node | Value |")
        lines.append("|---|---:|")
        bottom5 = df[col].sort_values(ascending=True).head(10)
        for idx, val in bottom5.items():
            lines.append(f"| `{str(idx)}` | {val:.4f} |")
        lines.append("")

    # Overview image
    lines.append(f"![Detailed overview]({detailed_overview_path.name})")

    with open(outfile, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"Wrote Markdown report to: {outfile}")

    p = Path(outfile)
    body = markdown.markdown(p.read_text(encoding="utf-8"), extensions=["extra", "toc", "tables"])

    html = f"""<!doctype html>
    <html>
    <head>
      <meta charset="utf-8">
      <style>
        @page {{ size: A4; margin: 20mm; }}
        h3 {{ page-break-before: always; }}
        body {{ font-size: 12px; line-height: 1.4; }}
        img {{ display: block; width: 100%; height: auto; max-width: 100%; }}
      </style>
    </head>
    <body>
    {body}
    </body>
    </html>"""

    weasyprint.HTML(string=html, base_url=str(p.parent)).write_pdf(outfile.with_suffix(".pdf"))
    print(f"Wrote PDF report to: {outfile.with_suffix('.pdf')}")

if __name__ == "__main__":
    main(
        data_path= paths.sample_derivatives / "dist-euclidean_ttype-percentage_tvalue-15_k-4_ninits-50_activation-pos_n-34"
    )
