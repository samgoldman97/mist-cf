# Predict mist_cf models on the test splits
mkdir results/mist_cf_predict_sirius/

python src/mist_cf/mist_cf_score/predict_mgf.py \
    --id-key FEATURE_ID \
    --gpu \
    --num-workers 32 \
    --batch-size 8 \
    --save-dir results/mist_cf_predict_sirius/mist_cf_1/ \
    --mgf-file data/nist_canopus/split_1_test.mgf \
    --checkpoint-pth results/mist_cf_nist/split_1_with_nist/version_0/best.ckpt \
    --fast-model results/fast_filter/split/version_0/best.ckpt \
    --fast-num 256 \
    --decomp-ppm 10 \
    --decomp-filter "RDBE" 

python src/mist_cf/mist_cf_score/predict_mgf.py \
    --id-key FEATURE_ID \
    --gpu \
    --num-workers 32 \
    --batch-size 8 \
    --save-dir results/mist_cf_predict_sirius/mist_cf_2/ \
    --mgf-file data/nist_canopus/split_2_test.mgf \
    --checkpoint-pth results/mist_cf_nist/split_2_with_nist/version_0/best.ckpt \
    --fast-model results/fast_filter/split/version_0/best.ckpt \
    --fast-num 256 \
    --decomp-ppm 10 \
    --decomp-filter "RDBE" 

CUDA_VISIBLE_DEVICES=3 python src/mist_cf/mist_cf_score/predict_mgf.py \
    --id-key FEATURE_ID \
    --gpu \
    --num-workers 32 \
    --batch-size 8 \
    --save-dir results/mist_cf_predict_sirius/mist_cf_3/ \
    --mgf-file data/nist_canopus/split_3_test.mgf \
    --checkpoint-pth results/mist_cf_nist/split_3_with_nist/version_0/best.ckpt \
    --fast-model results/fast_filter/split/version_0/best.ckpt \
    --fast-num 256 \
    --decomp-ppm 10 \
    --decomp-filter "RDBE" 
