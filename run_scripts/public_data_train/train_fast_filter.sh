CUDA_VISIBLE_DEVICES=0 python3 src/mist_cf/fast_form_score/train.py \
--gpu \
--save-dir 'split' \
--seed 1 \
--num-workers 8 \
--dataset-file 'data/biomols/biomols_with_decoys.txt' \
--split-file 'data/biomols/biomols_with_decoys_split.tsv' \
--batch-size 64 \
--max-decoy 32 \
--max-epochs 200 \
--learning-rate 0.00036 \
--lr-decay-frac 0.86425 \
--weight-decay 0 \
--layers 3 \
--dropout 0.1 \
--hidden-size 256 \
--form-encoder 'abs-sines' \
--save-dir results/public_fast_filter/split
