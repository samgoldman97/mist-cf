# Run all preprocessing in serial

# Make splits
# Wrap in if false  clause
#python3 preprocessing/01_create_split_file.py --data-dir data/nist_canopus --label-file data/nist_canopus/labels.tsv --val-frac 0.1 --test-frac 0.2 --seed 1
#python3 preprocessing/01_create_split_file.py --data-dir data/nist_canopus --label-file data/nist_canopus/labels.tsv --val-frac 0.1 --test-frac 0.2 --seed 2
#python3 preprocessing/01_create_split_file.py --data-dir data/nist_canopus --label-file data/nist_canopus/labels.tsv --val-frac 0.1 --test-frac 0.2 --seed 3

# Create decoy label
# These use fast filter now
python3 preprocessing/02_create_decoy_label.py --decomp-filter COMMON --label-file data/nist_canopus/labels.tsv --data-dir data/nist_canopus --max-decoy 256 --sample-strat fast_filter --resample-precursor-mz --fast-model results/fast_filter/split/version_0/best.ckpt --num-workers 32 
python3 preprocessing/02_create_decoy_label.py --decomp-filter RDBE --label-file data/nist_canopus/labels.tsv --data-dir data/nist_canopus --max-decoy 10000000 --sample-strat normalized_inverse  --resample-precursor-mz --num-workers 32 


# Create pred labels common
python3 preprocessing/03_create_pred_label.py --decoy-label data/nist_canopus/decoy_labels/decoy_label_COMMON.tsv --split-file data/nist_canopus/splits/split_1.tsv --data-dir data/nist_canopus/
python3 preprocessing/03_create_pred_label.py --decoy-label data/nist_canopus/decoy_labels/decoy_label_COMMON.tsv --split-file data/nist_canopus/splits/split_2.tsv --data-dir data/nist_canopus/
python3 preprocessing/03_create_pred_label.py --decoy-label data/nist_canopus/decoy_labels/decoy_label_COMMON.tsv --split-file data/nist_canopus/splits/split_3.tsv --data-dir data/nist_canopus/
#
#
## Create pred labels rdbe
python3 preprocessing/03_create_pred_label.py --decoy-label data/nist_canopus/decoy_labels/decoy_label_RDBE.tsv --split-file data/nist_canopus/splits/split_1.tsv --data-dir data/nist_canopus/
python3 preprocessing/03_create_pred_label.py --decoy-label data/nist_canopus/decoy_labels/decoy_label_RDBE.tsv --split-file data/nist_canopus/splits/split_2.tsv --data-dir data/nist_canopus/
python3 preprocessing/03_create_pred_label.py --decoy-label data/nist_canopus/decoy_labels/decoy_label_RDBE.tsv --split-file data/nist_canopus/splits/split_3.tsv --data-dir data/nist_canopus/
#
#
## Create pred labels common hyperopt
python3 preprocessing/03_create_pred_label.py --decoy-label data/nist_canopus/decoy_labels/decoy_label_COMMON.tsv --split-file data/nist_canopus/splits/split_1_hyperopt_10000.tsv --data-dir data/nist_canopus/
#
## Create pred labels common with nist
python3 preprocessing/03_create_pred_label.py --decoy-label data/nist_canopus/decoy_labels/decoy_label_COMMON.tsv --split-file data/nist_canopus/splits/split_1_with_nist.tsv --data-dir data/nist_canopus/
python3 preprocessing/03_create_pred_label.py --decoy-label data/nist_canopus/decoy_labels/decoy_label_COMMON.tsv --split-file data/nist_canopus/splits/split_2_with_nist.tsv  --data-dir data/nist_canopus/
python3 preprocessing/03_create_pred_label.py --decoy-label data/nist_canopus/decoy_labels/decoy_label_COMMON.tsv --split-file data/nist_canopus/splits/split_3_with_nist.tsv --data-dir data/nist_canopus/

#
## Create pred labels rdbe with nist
python3 preprocessing/03_create_pred_label.py --decoy-label data/nist_canopus/decoy_labels/decoy_label_RDBE.tsv --split-file data/nist_canopus/splits/split_1_with_nist.tsv --data-dir data/nist_canopus/
python3 preprocessing/03_create_pred_label.py --decoy-label data/nist_canopus/decoy_labels/decoy_label_RDBE.tsv --split-file data/nist_canopus/splits/split_2_with_nist.tsv --data-dir data/nist_canopus/
python3 preprocessing/03_create_pred_label.py --decoy-label data/nist_canopus/decoy_labels/decoy_label_RDBE.tsv --split-file data/nist_canopus/splits/split_3_with_nist.tsv --data-dir data/nist_canopus/

## Create subformula assign
python preprocessing/04_create_subformulae_assignment.py \
    --data-dir data/nist_canopus \
    --decoy-label data/nist_canopus/decoy_labels/decoy_label_COMMON.tsv \
    --max-formulae 20 \
    --inten-thresh 0.003 \
    --mass-diff-thresh 15 \
    --mass-diff-type ppm \
    #--debug

# Get mgf files for export
python preprocessing/05_sirius_to_mgf.py --split-file data/nist_canopus/splits/split_1.tsv --save-name data/nist_canopus/split_1_test_debug.mgf --ms-file-dir data/nist_canopus/spec_files/ --debug
python preprocessing/05_sirius_to_mgf.py --split-file data/nist_canopus/splits/split_1.tsv --save-name data/nist_canopus/split_1_test.mgf --ms-file-dir data/nist_canopus/spec_files/
python preprocessing/05_sirius_to_mgf.py --split-file data/nist_canopus/splits/split_2.tsv --save-name data/nist_canopus/split_2_test.mgf --ms-file-dir data/nist_canopus/spec_files/
python preprocessing/05_sirius_to_mgf.py --split-file data/nist_canopus/splits/split_3.tsv --save-name data/nist_canopus/split_3_test.mgf --ms-file-dir data/nist_canopus/spec_files/