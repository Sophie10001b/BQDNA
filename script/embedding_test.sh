ckpt_path="result/finetune"
logger_project="VQ-Embedding_test"

batch_size=64

latent_size=128
codebook_size=16384
downsample_rate=5
num_heads=2
vq_type="optvq"

for seed in 17
do
    vq_path="/root/autodl-fs/pretrain/VQ/8192/16/final"
    
    data_path="data/finetune/GUE"
    dataset_name="GUE"

    # for data in prom_core_all
    # do
    #     python embedding_test.py \
    #         --seed ${seed} \
    #         --data_path "${data_path}" \
    #         --data_name "prom/${data}" \
    #         --dataset_name "${dataset_name}" \
    #         --ckpt_path "${ckpt_path}" \
    #         --vq_path "${vq_path}" \
    #         --batch_size ${batch_size} \
    #         --num_workers 4 \
    #         --logger_project "${logger_project}" \
    #         --latent_size ${latent_size} \
    #         --codebook_size ${codebook_size} \
    #         --downsample_rate ${downsample_rate} \
    #         --num_heads ${num_heads} \
    #         --vq_type "${vq_type}"
    # done

    data_path="data/finetune/embedding"
    dataset_name="embedding"

    # for data in NT_biotype
    # do
    #     python embedding_test.py \
    #         --seed ${seed} \
    #         --data_path "${data_path}" \
    #         --data_name "${data}" \
    #         --dataset_name "${dataset_name}" \
    #         --ckpt_path "${ckpt_path}" \
    #         --vq_path "${vq_path}" \
    #         --batch_size ${batch_size} \
    #         --num_workers 4 \
    #         --logger_project "${logger_project}" \
    #         --latent_size ${latent_size} \
    #         --codebook_size ${codebook_size} \
    #         --downsample_rate ${downsample_rate} \
    #         --num_heads ${num_heads} \
    #         --vq_type "${vq_type}"
    # done

    # for data in species_1024_10000_1000_1000
    # do
    #     python embedding_test.py \
    #         --seed ${seed} \
    #         --data_path "${data_path}/species" \
    #         --data_name "${data}" \
    #         --dataset_name "${dataset_name}" \
    #         --ckpt_path "${ckpt_path}" \
    #         --vq_path "${vq_path}" \
    #         --batch_size 16 \
    #         --num_workers 4 \
    #         --logger_project "${logger_project}" \
    #         --latent_size ${latent_size} \
    #         --codebook_size ${codebook_size} \
    #         --downsample_rate ${downsample_rate} \
    #         --num_heads ${num_heads} \
    #         --vq_type "${vq_type}"
    # done

    for data in species_16384_10000_1000_1000
    do
        python embedding_test.py \
            --seed ${seed} \
            --data_path "${data_path}/species" \
            --data_name "${data}" \
            --dataset_name "${dataset_name}" \
            --ckpt_path "${ckpt_path}" \
            --vq_path "${vq_path}" \
            --batch_size 16 \
            --num_workers 4 \
            --logger_project "${logger_project}" \
            --latent_size ${latent_size} \
            --codebook_size ${codebook_size} \
            --downsample_rate ${downsample_rate} \
            --num_heads ${num_heads} \
            --vq_type "${vq_type}"
    done
done