set -x
set -e

NEOX_DIR=gpt-neox
# VOCAB_DIR=vocab-data/olmo-new/
MODEL_DIR=/data/dclm-baseline-1.0/models/olmo_240M_interference/

# hubble_v3_testset_copyright_seq_shuffle_lr0004_10k
declare -a joblist=("experiments/20250209_dclm_hubble_v3_testset_copyright_seq_shuffle_10k/olmo_240M,hubble_v3_testset_copyright_seq_shuffle_lr0004_10k"
)

for i in "${joblist[@]}"
do
    IFS=',' read CONFIG_DIR exp_name <<< "${i}"
    config_file=${CONFIG_DIR}/240M_export.yml

    neox_out_dir="${MODEL_DIR}/${exp_name}/global_step10000"
    hf_out_dir="${neox_out_dir}/hf_model"

    python $NEOX_DIR/tools/ckpts/convert_neox_to_hf_UPDATED.py \
            --input_dir $neox_out_dir \
            --config_file $config_file \
            --output_dir $hf_out_dir \
            --precision auto \
            --vocab-is-hf-tokenizer \
            --architecture llama
    
    nfs_out_dir="/home/ameya/HubbleSuite/models/olmo_240M_interference/${exp_name}/global_step10000/"
    mkdir -p ${nfs_out_dir}
    cp -rv ${hf_out_dir}/ ${nfs_out_dir}/
    echo "+++++++++++++++++++++++++++++++"
done
