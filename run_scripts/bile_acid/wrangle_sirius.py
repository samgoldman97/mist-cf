""" Process sirius results """
from pathlib import Path
import pandas as pd
from tqdm import tqdm

res_dirs = ["results/sirius_predict_bile_acid/sirius_1",]

for res_dir in res_dirs:
    res_dir = Path(res_dir)
    best_df = res_dir / "formula_identifications.tsv"
    best_df = pd.read_csv(best_df, sep="\t")

    id_to_mass = dict(best_df[['id', 'ionMass']].values)

    out_df = []

    for cand in tqdm(Path(res_dir).rglob("formula_candidates.tsv")):
        df = pd.read_csv(cand, sep="\t")

        spec_str = cand.parent.name
        spec = spec_str.split("ScanNumber")[-1]

        # replace space and strip
        df['cand_ion'] = df['adduct'].str.replace(" ", "")
        df['scores'] = df['SiriusScore']
        df['cand_form'] = df['molecularFormula']
        df['parentmasses'] = id_to_mass.get(spec_str, -1)
        df['spec'] = spec
        out_df.append(df[["spec", "cand_form", "scores", "cand_ion", "parentmasses"]])
    # Concat df
    out_df = pd.concat(out_df).reset_index(drop=True)
    out_df.to_csv(Path(res_dir) / "formatted_output.tsv", 
                  sep="\t", index=False)
