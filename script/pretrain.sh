nproc=2
seed=17
data_path="data/pretrain/fungi"
ckpt_path="result/pretrain"
vq_path="/root/autodl-tmp/vq/result/pretrain/Pretrain/VQ/8192/16/2025-05-28 20:29:30/pytorch_model.bin"

max_steps=-1
max_epochs=1
save_steps=-1

max_seqlen=16384
max_token_per_batch=`expr 1048576 \* $nproc`
accumulate_grad_batches=1
num_preprocess_workers=48

max_lr=5e-4
min_lr=0

latent_size=256
codebook_size=8192
downsample_rate=16
num_heads=1
vq_type="optvq"

python pretrain.py \
    --data_path $data_path \
    --ckpt_path $ckpt_path \
    --vq_path "${vq_path}" \
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
    --latent_size $latent_size \
    --codebook_size $codebook_size \
    --downsample_rate $downsample_rate \
    --num_heads $num_heads \
    --vq_type "${vq_type}"