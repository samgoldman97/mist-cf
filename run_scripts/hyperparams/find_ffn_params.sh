dataset=nist_canopus
CUDA_VISIBLE_DEVICES=1,2,3 python src/mist_cf/ffn_score/ffn_hyperopt.py \
    --dataset-name $dataset \
    --split-file data/$dataset/splits/split_1_hyperopt_10000.tsv \
    --decoy-label data/$dataset/decoy_labels/decoy_label_COMMON.tsv \
    --save-dir results/hyperopt_ffn \
    --seed 1 \
    --num-workers 8 \
    --max-epochs 200 \
    --max-decoy 32 \
    --gpu \
    --cpus-per-trial 8 \
    --gpus-per-trial 0.5 \
    --num-h-samples 50 \
    --no-cls-mass-diff
