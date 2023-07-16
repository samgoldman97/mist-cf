dataset=nist_canopus
# TODO: Switch to more gpus available
CUDA_VISIBLE_DEVICES="0,2" python src/mist_cf/mist_cf_score/mist_cf_hyperopt.py \
    --dataset-name $dataset \
    --split-file data/$dataset/splits/split_1_hyperopt_10000.tsv \
    --save-dir results/hyperopt_mist_cf \
    --decoy-label data/$dataset/decoy_labels/decoy_label_COMMON.tsv \
    --subform-dir data/$dataset/subformulae/formulae_spec_decoy_label_COMMON/ \
    --no-cls-mass-diff \
    --max-subpeak 20 \
    --seed 1 \
    --num-workers 8 \
    --max-epochs 200 \
    --max-decoy 32 \
    --gpu \
    --cpus-per-trial 8 \
    --gpus-per-trial 1 \
    --num-h-samples 50 


## -- debug
