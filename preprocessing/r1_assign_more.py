## Create subformula assign
python preprocessing/04_create_subformulae_assignment.py \
    --data-dir data/nist_canopus \
    --decoy-label data/nist_canopus/decoy_labels/decoy_label_COMMON.tsv \
    --max-formulae 50 \
    --inten-thresh 0.003 \
    --mass-diff-thresh 15 \
    --mass-diff-type ppm \
    --out-name formulae_spec_decoy_label_COMMON_50 \
    --num-workers 64

    #--debug
