seed=17
data_path="data/pretrain"
ckpt_path="result/pretrain"

max_steps=-1
max_epochs=1
save_steps=-1

max_seqlen=2048
max_token_per_batch=262144
accumulate_grad_batches=2
num_preprocess_workers=24

max_lr=5e-4
min_lr=0

codebook_size=4096
downsample_rate=16
vq_type=""

python pretrain.py \
    --data_path $data_path \
    --ckpt_path $ckpt_path \
    --seed $seed \
    --max_steps $max_steps \
    --max_epochs $max_epochs \
    --save_steps $save_steps \
    --max_seqlen $max_seqlen \
    --max_token_per_batch $max_token_per_batch \
    --accumulate_grad_batches $accumulate_grad_batches \
    --num_preprocess_workers $num_preprocess_workers \
    --max_lr $max_lr \
    --min_lr $min_lr \
    --codebook_size $codebook_size \
    --downsample_rate $downsample_rate \
    --vq_type $vq_type