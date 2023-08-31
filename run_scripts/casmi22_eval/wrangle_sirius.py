""" Process sirius results """
from pathlib import Path
import pandas as pd
from tqdm import tqdm

res_dirs = [
    "results/sirius_predict_casmi22/sirius_1",
]


for res_dir in res_dirs:
    res_dir = Path(res_dir)
    struct_dir = res_dir.parent / f"{res_dir.name}_structure"
    struct_dir.mkdir(exist_ok=True)
    struct_df = res_dir / "compound_identifications.tsv"
    best_df = res_dir / "formula_identifications.tsv"
    best_df = pd.read_csv(best_df, sep="\t")

    id_to_mass = dict(best_df[["id", "ionMass"]].values)

    out_df = []
    out_df_struct = []

    for cand in tqdm(Path(res_dir).rglob("formula_candidates.tsv")):

        df = pd.read_csv(cand, sep="\t")

        spec_str = cand.parent.name
        spec = spec_str.split("_", 4)[-1]
        parentmass = id_to_mass.get(spec_str, -1)

        # replace space and strip
        df["cand_ion"] = df["adduct"].str.replace(" ", "")
        df["scores"] = df["SiriusScore"]
        df["cand_form"] = df["molecularFormula"]
        df["parentmasses"] = parentmass
        df["spec"] = spec
        out_df.append(df[["spec", "cand_form", "scores", "cand_ion", "parentmasses"]])

        struct_df_file = cand.parent / "structure_candidates.tsv"
        if struct_df_file.exists():
            struct_df = pd.read_csv(struct_df_file, sep="\t")
            # Sort by CSI:FingerIDScore (highest to lowest), then make unique by molecualr formula
            struct_df = struct_df.sort_values("CSI:FingerIDScore", ascending=False)

            # Drop duplicates
            struct_df = struct_df.drop_duplicates(
                subset="molecularFormula", keep="first"
            )

            struct_df["spec"] = spec
            struct_df["parentmasses"] = parentmass
            struct_df["cand_ion"] = struct_df["adduct"].str.replace(" ", "")
            struct_df["scores"] = struct_df["CSI:FingerIDScore"]
            struct_df["cand_form"] = struct_df["molecularFormula"]
            out_df_struct.append(
                struct_df[["spec", "cand_form", "scores", "cand_ion", "parentmasses"]]
            )

    # Export
    out_df = pd.concat(out_df).reset_index(drop=True)
    out_df.to_csv(res_dir / "formatted_output.tsv", sep="\t", index=False)

    out_df = pd.concat(out_df_struct).reset_index(drop=True)
    out_df.to_csv(struct_dir / "formatted_output.tsv", sep="\t", index=False)
