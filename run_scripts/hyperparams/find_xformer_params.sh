dataset=nist_canopus
CUDA_VISIBLE_DEVICES="1,3" python src/mist_cf/xformer_score/xformer_hyperopt.py \
    --dataset-name $dataset \
    --split-file data/$dataset/splits/split_1_hyperopt_10000.tsv \
    --decoy-label data/$dataset/decoy_labels/decoy_label_COMMON.tsv \
    --save-dir results/hyperopt_xformer \
    --no-cls-mass-diff \
    --seed 1 \
    --num-workers 8 \
    --max-epochs 200 \
    --max-decoy 32 \
    --gpu \
    --cpus-per-trial 8 \
    --gpus-per-trial 1 \
    --num-h-samples 50 
