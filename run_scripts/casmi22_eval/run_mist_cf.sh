# Predict mist_cf models on the test splits
mkdir results/mist_cf_predict_casmi22/
python src/mist_cf/mist_cf_score/predict_mgf.py \
    --id-key FEATURE_ID \
    --gpu \
    --num-workers 32 \
    --batch-size 8 \
    --save-dir results/mist_cf_predict_casmi22/ \
    --mgf-file data/casmi22/CASMI_processed.mgf \
    --checkpoint-pth results/mist_cf_nist/split_1_with_nist/version_0/best.ckpt \
    --fast-model results/fast_filter/split/version_0/best.ckpt \
    --fast-num 256 \
    --instrument-override "Orbitrap (LCMS)" \
    --decomp-ppm 5 \
    --decomp-filter "RDBE"

    #--decomp-filter "COMMON" 
    #--debug
    #--mgf-file data/nist_canopus/split_1_test_debug.mgf \
