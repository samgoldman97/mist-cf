"""predict_mgf.py

Make predictions with trained model from scratch

"""
import logging
import yaml
import argparse
from pathlib import Path
from collections import defaultdict
import time

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl

import mist_cf.common as common
import mist_cf.decomp as decomp
from mist_cf.mist_cf_score import mist_cf_data, mist_cf_model
from mist_cf.fast_form_score import fast_form_data, fast_form_model


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--debug", default=False, action="store_true", help="Debug flag"
    )
    parser.add_argument(
        "--id-key", default="FEATURE_ID", 
        action="store", help="The key to use for the id"
    )
    parser.add_argument(
        "--instrument-override", default=None, 
        action="store", help="Optional instrument override string"
    )
    parser.add_argument(
        "--gpu", default=False, action="store_true", help="Use GPU flag"
    )
    parser.add_argument(
        "--num-workers",
        default=0,
        action="store",
        type=int,
        help="Set number of available CPUs",
    )
    parser.add_argument(
        "--batch-size",
        default=8,
        action="store",
        type=int,
        help="Set number of batch size",
    )
    parser.add_argument("--save-dir", required=True, help="Path to save directory")
    parser.add_argument(
        "--mgf-file",
        default="data/mills/gnps_fbn_export_debug.mgf",
        help="Path to prediction label",
    )
    parser.add_argument(
        "--checkpoint-pth",
        default="results/2022_09_22_ffn_train/version_0/epoch=20-val_loss=0.46.ckpt",
        help="Path to check point file",
    )
    parser.add_argument(
        "--decomp-filter",
        type=str,
        default="RDBE",
        help="Filter used for decoy generation.",
    )

    parser.add_argument(
        "--fast-model",
        type=str,
        default=None,
        help="Name of fast filter model to load"
    )
    parser.add_argument(
        "--fast-num",
        type=int,
        default=None,
        help="Num of formulae to keep per spec with fast formulae model"
    )
    parser.add_argument(
        "--decomp-ppm",
        type=int,
        default=5,
        help="relative mass error for candidate space generation"
    )
    return parser.parse_args()


def gen_cand_space(spec_to_parent: dict, decomp_filter: str,
                   save_out: Path = None, debug : bool= False, 
                   ppm: int = 5, ions=common.ION_LST,
                   num_workers=16,) -> pd.DataFrame:
    """gen_cand_space.

    Args:
        spec_to_parent (dict): spec_to_parent
        decomp_filter (str): decomp_filter
        save_out (Path): save_out
        debug (bool): debug

    Returns:
        pd.DataFrame:
    """

    specs, precursor_mz = zip(*list(spec_to_parent.items()))

    all_out_dicts = defaultdict(lambda: set())
    for ion in ions:
        # equation: parentmass = decoy formula + decoy ionization
        decoy_masses = [
            (parentmass - common.ion_to_mass[ion])
            for parentmass in precursor_mz
        ]
        decoy_masses = decomp.get_rounded_masses(decoy_masses)
        spec2mass = dict(zip(specs, decoy_masses))

        # Let's replace decomp.run_sirius
        # Switch to ppm=10 for speed
        out_dict = decomp.run_sirius(decoy_masses, filter_=decomp_filter,
                                     ppm=ppm, cores=num_workers) 
        out_dict = {k: {(ion, vv) for vv in v} for k, v in out_dict.items()}

        # Update the existing all_out_dicts with the new out_dict
        for spec, mass in spec2mass.items():
            # Add out_dict to all_out dicts
            all_out_dicts[spec].update(out_dict.get(mass, {}))


    all_ions = [",".join([ion for ion, form in all_out_dicts[i]]) for i in specs]
    all_forms = [",".join([form for ion, form in all_out_dicts[i]]) for i in specs]

    data = {
        "spec": specs,
        "cand_form": all_forms,
        "cand_ion": all_ions,
        "parentmass": precursor_mz,
    }
    output_df = pd.DataFrame.from_dict(data)

    # Unroll the data frame s.t. each row is a single ion 
    new_dict = []
    for _, row in output_df.iterrows():
        for ion, form in zip(row['cand_ion'].split(","), row['cand_form'].split(",")):
            new_dict.append({"spec": row['spec'], "cand_ion": ion, "cand_form": form, "parentmass": row['parentmass']})
    output_df = pd.DataFrame.from_dict(new_dict)

    if save_out is not None:
        output_df.to_csv(save_out, sep="\t", index=None)
    return output_df


def predict():
    args = get_args()
    kwargs = args.__dict__
    debug = kwargs["debug"]
    mgf_path = Path(kwargs['mgf_file'])
    mass_diff_thresh = 15
    instrument_key = 'INSTRUMENT'
    ms1_key = "PEPMASS"
    id_key = kwargs['id_key']
    instrument_override = kwargs['instrument_override']

    # For fast filtering
    fast_model = kwargs['fast_model']
    fast_num = kwargs['fast_num']

    gpu = kwargs["gpu"]
    device = torch.device("cuda") if gpu else torch.device("cpu")

    save_dir = Path(kwargs['save_dir'])
    save_dir.mkdir(exist_ok=True)
    save_name = "formatted_output.tsv"
    save_name = save_dir / save_name

    common.setup_logger(
        kwargs["save_dir"], log_name=f"mist_cf_pred.log", debug=kwargs["debug"]
    )
    pl.utilities.seed.seed_everything(kwargs.get("seed"))

    subform_dir = save_dir / "subform_assigns"
    subform_dir.mkdir(exist_ok=True)

    # Dump args
    yaml_args = yaml.dump(kwargs, indent=2, default_flow_style=False)
    logging.info(f"Args:\n{yaml_args}")
    with open(Path(kwargs["save_dir"]) / "args.yaml", "w") as fp:
        fp.write(yaml_args)

    max_num = None
    num_workers = kwargs['num_workers']
    num_workers = 0 if num_workers == 1 else num_workers
    if debug:
        max_num = 10
        num_workers = 0


    # Get corresponding prediction label
    # Create model and load
    best_checkpoint = kwargs["checkpoint_pth"]

    # Load from checkpoint
    model = mist_cf_model.MistNet.load_from_checkpoint(best_checkpoint)
    logging.info(f"Loaded model with from {best_checkpoint}")
    max_subpeak = model.max_subpeak

    logging.info("Loading mgf spectra")
    specs = common.parse_spectra_mgf(mgf_path, max_num=max_num)
    metas, specs = zip(*specs)
    specs = [[spec[0][1]] for spec in specs]
    specs = [common.merge_spec_tuples(i, parent_mass=float(meta[ms1_key])) for meta, i in zip(metas, specs)]
    specs = [common.max_thresh_spec(i, max_peaks=model.max_subpeak,
                                     inten_thresh=0.003) for i in specs]
    spec_ids = [i[id_key] for i in metas]
    parent_masses = [float(i[ms1_key]) for i in metas]
    instruments = [i[instrument_key] if instrument_key in i else "Unknown (LCMS)" for i in metas]

    # subsetting mgf spectra
    id_to_meta = dict(zip(spec_ids, metas))
    id_to_ms1 =  dict(zip(spec_ids, parent_masses))
    id_to_ms2 = dict(zip(spec_ids, specs))
    id_to_instrument = dict(zip(spec_ids, instruments))
    ions = common.ION_LST if not debug else ['[M+H]+']

    # Generate candidate space --> save pred file (using PrecursorMZ)
    save_cands = save_dir / "pred_labels.tsv"
    label_df = gen_cand_space(id_to_ms1,
                              kwargs['decomp_filter'],
                              save_out=save_cands,
                              debug=debug,
                              ppm=kwargs['decomp_ppm'],
                              ions=ions,
                              num_workers=num_workers)

    save_cands_filter = save_dir / "pred_labels_filter.tsv"
    if fast_num is not None and fast_model is not None:
        logging.info(f"Fast filtering down to {fast_num} cands per spec")
        new_df = fast_form_model.fast_filter_df(label_df=label_df, fast_num=fast_num,
                                                fast_model=fast_model, device=device,
                                                num_workers=num_workers)
        label_df = new_df

    # Add in instrument
    instruments = [id_to_instrument[str(spec)] for spec in label_df['spec'].values]
    label_df['instrument'] = instruments
    if instrument_override is not None:
        label_df['instrument'] = instrument_override


    label_df.to_csv(save_cands_filter, sep="\t", index=None)

    # Note: Consider abstracting code below into separate functions s.t. we can
    # iterate over label df in chunks and not save everything to disk if we want

    # Begin subformulae assignment
    # Convert df into spec to forms and spec to ions
    spec_to_entries = defaultdict(lambda: {"forms": [], "ions": []})
    for _, row in label_df.iterrows():
        row_key = str(row['spec'])
        spec_to_entries[row_key]['forms'].append(row['cand_form'])
        spec_to_entries[row_key]['ions'].append(row['cand_ion'])

    all_entries = []
    for spec_id, ms2 in tqdm(id_to_ms2.items()):
        forms = spec_to_entries[spec_id]['forms']
        ions = spec_to_entries[spec_id]['ions']
        mass_diff_thresh = common.get_instr_tol(id_to_instrument[spec_id])
        new_entries = [
            {"spec": ms2, "mass_diff_type": "ppm", "spec_name": spec_id,
             "mass_diff_thresh": mass_diff_thresh, "form": form, "ion_type": ion}
            for form, ion in zip(forms, ions)
        ]
        new_item = {"spec_name": spec_id, "export_dicts": new_entries,
                    "output_dir": subform_dir}
        all_entries.append(new_item)

    logging.info(f"Assigning subformula")
    export_wrapper = lambda x: common.assign_single_spec(**x)
    if num_workers == 0 or debug:
        [export_wrapper(i) for i in tqdm(all_entries)]
    else:
        common.chunked_parallel(
            all_entries, export_wrapper, chunks=100, max_cpu=num_workers
        )

    # Create dataset
    # Define num bins
    pred_dataset = mist_cf_data.PredDataset(
        label_df,
        subform_dir=subform_dir,
        num_workers=num_workers,
        max_subpeak=model.max_subpeak,
        ablate_cls_error=not model.cls_mass_diff
    )
    # Define dataloaders
    collate_fn = pred_dataset.get_collate_fn()
    pred_loader = DataLoader(
        pred_dataset,
        num_workers=kwargs["num_workers"],
        collate_fn=collate_fn,
        shuffle=False,
        batch_size=kwargs["batch_size"],
    )

    model.eval()
    model = model.to(device)

    logging.info(f"Predicting spectra with mist-cf")
    out_names, out_forms, out_scores, out_ions, out_parentmasses = [], [], [], [], []
    with torch.no_grad():
        for batch in pred_loader:
            peak_types, form_vec, ion_vec, instrument_vec, intens, rel_mass_diffs, num_peaks = (
                batch["types"],
                batch["form_vec"],
                batch["ion_vec"],
                batch["instrument_vec"],
                batch["intens"],
                batch["rel_mass_diffs"],
                batch["num_peaks"],
            )
            peak_types = peak_types.to(device)
            form_vec = form_vec.to(device)
            ion_vec = ion_vec.to(device)
            instrument_vec = instrument_vec.to(device)
            intens = intens.to(device)
            rel_mass_diffs = rel_mass_diffs.to(device)
            num_peaks = num_peaks.to(device)

            model_outs = model.forward(
                num_peaks, peak_types, form_vec, ion_vec, instrument_vec, intens, rel_mass_diffs
            )
            # ex_inds = batch['example_inds'].long()
            # num_inputs = batch['num_inputs']

            actual_forms = batch["str_forms"]
            actual_ions = batch["str_ions"]
            parentmasses = batch["parentmasses"]
            scores = model_outs.squeeze().cpu().numpy()
            # names = np.array(batch['names'])[batch['example_inds'].cpu().numpy()]
            names = np.array(batch["names"])

            out_names.extend(names)
            out_scores.extend(scores)
            out_forms.extend(actual_forms)
            out_ions.extend(actual_ions)
            out_parentmasses.extend(parentmasses)

        output = {
            "names": out_names,
            "forms": out_forms,
            "scores": out_scores,
            "ions": out_ions,
            "parentmasses": out_parentmasses,
        }

        logging.info("Exporting")
        out_df = pd.DataFrame(output)
        # Sort by names then scores
        out_df = out_df.sort_values(by=["names", "scores"], ascending=False)


        out_df = out_df.rename(
            columns={"names": "spec", "forms": "cand_form", "ions": "cand_ion"}
        )
        out_df.to_csv(save_name, sep="\t", index=None)


if __name__ == "__main__":
    start_time = time.time()
    predict()
    end_time = time.time()
    logging.info(f"Program finished in: {end_time - start_time} seconds")
