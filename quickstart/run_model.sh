fast_filter_model="quickstart/models/fast_filter_best.ckpt"
mist_cf_model="quickstart/models/mist_cf_best.ckpt"
out_dir="quickstart/mist_cf_out/"
mgf_file="data/demo_specs.mgf"

mkdir $out_dir

python src/mist_cf/mist_cf_score/predict_mgf.py \
    --id-key FEATURE_ID \
    --num-workers 0 \
    --batch-size 8 \
    --save-dir $out_dir \
    --mgf-file $mgf_file \
    --checkpoint-pth $mist_cf_model \
    --fast-model $fast_filter_model \
    --fast-num 256 \
    --instrument-override "Orbitrap (LCMS)" \
    --decomp-ppm 5 \
    --decomp-filter "RDBE" \

    # --gpu 
