data_path="data/finetune/GUE"
ckpt_path="result/finetune"
logger_project="VQTransformer|finetune"
dataset_name="GUE"

batch_size=64
max_lr=1e-4
min_lr=0

latent_size=256
codebook_size=8192
downsample_rate=16
num_heads=1
vq_type="optvq"

for seed in 17
do
    pretrained_ckpt_path="/root/autodl-fs/pretrain/VQTransformer/CLM/final"
    vq_path="/root/autodl-fs/pretrain/VQ/8192/16/final"
    
    # H3 H3K14ac H3K36me3 H3K4me1 H3K4me2 H3K4me3 H3K79me3 H3K9ac H4 H4ac
    for data in H3 H3K14ac H3K36me3 H3K4me1 H3K4me2 H3K4me3 H3K79me3 H3K9ac H4 H4ac
    do
        python finetune.py \
            --seed ${seed} \
            --data_path "${data_path}" \
            --data_name "EMP/${data}" \
            --ckpt_path "${ckpt_path}" \
            --pretrained_ckpt_path "${pretrained_ckpt_path}" \
            --vq_path "${vq_path}" \
            --batch_size ${batch_size} \
            --eval_batch_size ${batch_size} \
            --accumulate_grad_batches 1 \
            --num_workers 4 \
            --max_steps 5000 \
            --eval_start 200 \
            --eval_step 200 \
            --precision "bf16-mixed" \
            --max_lr ${max_lr} \
            --min_lr ${min_lr} \
            --warmup_ratio 0.1 \
            --core_metric "MCC" \
            --logger_project "${logger_project}" \
            --dataset_name "${dataset_name}" \
            --latent_size ${latent_size} \
            --codebook_size ${codebook_size} \
            --downsample_rate ${downsample_rate} \
            --num_heads ${num_heads} \
            --vq_type "${vq_type}"
    done

    for data in 0 1 2 3 4
    do
        python finetune.py \
            --seed ${seed} \
            --data_path "${data_path}" \
            --data_name "mouse/${data}" \
            --ckpt_path "${ckpt_path}" \
            --pretrained_ckpt_path "${pretrained_ckpt_path}" \
            --vq_path "${vq_path}" \
            --batch_size ${batch_size} \
            --eval_batch_size ${batch_size} \
            --accumulate_grad_batches 1 \
            --num_workers 4 \
            --max_steps 5000 \
            --eval_start 200 \
            --eval_step 200 \
            --precision "bf16-mixed" \
            --max_lr ${max_lr} \
            --min_lr ${min_lr} \
            --warmup_ratio 0.1 \
            --core_metric "MCC" \
            --logger_project "${logger_project}" \
            --dataset_name "${dataset_name}" \
            --latent_size ${latent_size} \
            --codebook_size ${codebook_size} \
            --downsample_rate ${downsample_rate} \
            --num_heads ${num_heads} \
            --vq_type "${vq_type}"
    done

    for data in covid
    do
        python finetune.py \
            --seed ${seed} \
            --data_path "${data_path}" \
            --data_name "virus/${data}" \
            --ckpt_path "${ckpt_path}" \
            --pretrained_ckpt_path "${pretrained_ckpt_path}" \
            --vq_path "${vq_path}" \
            --batch_size ${batch_size} \
            --eval_batch_size ${batch_size} \
            --accumulate_grad_batches 1 \
            --num_workers 4 \
            --max_steps 10000 \
            --eval_start 200 \
            --eval_step 200 \
            --precision "bf16-mixed" \
            --max_lr ${max_lr} \
            --min_lr ${min_lr} \
            --warmup_ratio 0.1 \
            --core_metric "MCC" \
            --logger_project "${logger_project}" \
            --dataset_name "${dataset_name}" \
            --latent_size ${latent_size} \
            --codebook_size ${codebook_size} \
            --downsample_rate ${downsample_rate} \
            --num_heads ${num_heads} \
            --vq_type "${vq_type}"
    done

    for data in 0 1 2 3 4
    do
        python finetune.py \
            --seed ${seed} \
            --data_path "${data_path}" \
            --data_name "tf/${data}" \
            --ckpt_path "${ckpt_path}" \
            --pretrained_ckpt_path "${pretrained_ckpt_path}" \
            --vq_path "${vq_path}" \
            --batch_size ${batch_size} \
            --eval_batch_size ${batch_size} \
            --accumulate_grad_batches 1 \
            --num_workers 4 \
            --max_steps 5000 \
            --eval_start 200 \
            --eval_step 200 \
            --precision "bf16-mixed" \
            --max_lr ${max_lr} \
            --min_lr ${min_lr} \
            --warmup_ratio 0.1 \
            --core_metric "MCC" \
            --logger_project "${logger_project}" \
            --dataset_name "${dataset_name}" \
            --latent_size ${latent_size} \
            --codebook_size ${codebook_size} \
            --downsample_rate ${downsample_rate} \
            --num_heads ${num_heads} \
            --vq_type "${vq_type}"
    done

    for data in prom_300_all prom_300_tata prom_300_notata prom_core_all prom_core_tata prom_core_notata
    do
        python finetune.py \
            --seed ${seed} \
            --data_path "${data_path}" \
            --data_name "prom/${data}" \
            --ckpt_path "${ckpt_path}" \
            --pretrained_ckpt_path "${pretrained_ckpt_path}" \
            --vq_path "${vq_path}" \
            --batch_size ${batch_size} \
            --eval_batch_size ${batch_size} \
            --accumulate_grad_batches 1 \
            --num_workers 4 \
            --max_steps 5000 \
            --eval_start 200 \
            --eval_step 200 \
            --precision "bf16-mixed" \
            --max_lr ${max_lr} \
            --min_lr ${min_lr} \
            --warmup_ratio 0.1 \
            --core_metric "MCC" \
            --logger_project "${logger_project}" \
            --dataset_name "${dataset_name}" \
            --latent_size ${latent_size} \
            --codebook_size ${codebook_size} \
            --downsample_rate ${downsample_rate} \
            --num_heads ${num_heads} \
            --vq_type "${vq_type}"
    done

    for data in splice
    do
        python finetune.py \
            --seed ${seed} \
            --data_path "${data_path}" \
            --data_name "${data}" \
            --ckpt_path "${ckpt_path}" \
            --pretrained_ckpt_path "${pretrained_ckpt_path}" \
            --vq_path "${vq_path}" \
            --batch_size ${batch_size} \
            --eval_batch_size ${batch_size} \
            --accumulate_grad_batches 1 \
            --num_workers 4 \
            --max_steps 5000 \
            --eval_start 200 \
            --eval_step 200 \
            --precision "bf16-mixed" \
            --max_lr ${max_lr} \
            --min_lr ${min_lr} \
            --warmup_ratio 0.1 \
            --core_metric "MCC" \
            --logger_project "${logger_project}" \
            --dataset_name "${dataset_name}" \
            --latent_size ${latent_size} \
            --codebook_size ${codebook_size} \
            --downsample_rate ${downsample_rate} \
            --num_heads ${num_heads} \
            --vq_type "${vq_type}"
    done
done