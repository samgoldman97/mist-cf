# Filter smiles 
python preprocessing/biomols_scripts/01_filter_smiles.py --smiles-list data/biomols/biomols.txt --smiles-out data/biomols/biomols_filter.txt 


# Extract forms
python preprocessing/biomols_scripts/02_extract_formulae.py --smiles-list data/biomols/biomols_filter.txt --form-out data/biomols/biomols_filter_formulae.txt 


# Find decoys
python preprocessing/biomols_scripts/03_create_formulae_decoys.py --num-decoys 256  --out data/biomols/biomols_with_decoys.txt --decomp-filter RDBE


# Create split to exclude antyhing that appeaers in the other dataset
python preprocessing/biomols_scripts/04_create_formulae_split.py --formula-decoy-file data/biomols/biomols_with_decoys.txt --exclude-labels data/nist_canopus/labels.tsv --out data/biomols/biomols_with_decoys_split.tsv