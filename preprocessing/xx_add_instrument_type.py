from pathlib import Path
import argparse
import numpy as np
import pandas as pd
from mist_cf import common
from collections import Counter
import json

# Adding dataset
data_folder = Path("data/nist_canopus")
raw_label = data_folder / "labels.tsv"
spec_folder = data_folder / "spec_files"

raw_df = pd.read_csv(raw_label, sep='\t')
spec_to_instrument=dict()

def get_spec_instrument(spec_file):
    meta, tuples = common.parse_spectra(spec_file)

    # Try multiple
    if 'nist' in spec_file.stem:
        ins_string = meta.get('INSTRUMENT', "Unknown (LC/MS)")
    else:
        ins_string = meta.get('instrumentation', "Unknown (LC/MS)")
    return ins_string.strip()


spec_files = [spec_folder / f'{spec}.ms' for spec in raw_df['spec'].values]
all_instruments = common.chunked_parallel(spec_files, get_spec_instrument)
raw_df['instrument'] = all_instruments
instrument = Counter(all_instruments)
print(json.dumps(instrument, indent=2))

raw_df.to_csv(raw_label, sep="\t", index=None)
