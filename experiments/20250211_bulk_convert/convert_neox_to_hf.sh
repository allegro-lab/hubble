set -x
set -e

NEOX_DIR=gpt-neox
# VOCAB_DIR=vocab-data/olmo-new/
MODEL_DIR=/data/dclm-baseline-1.0/models/olmo_240M_interference/

# standard_lr0004_10k
# hubble_v3_copyright_privacy_seq_shuffle_lr0004_10k
declare -a joblist=("experiments/20250209_dclm_standard_higer_lr_10k/olmo_240M,standard_lr0004_10k"
                    "experiments/20250209_dclm_hubble_v3_copyright_privacy_seq_shuffle_10k/olmo_240M,hubble_v3_copyright_privacy_seq_shuffle_lr0004_10k"
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

# # dclm_standard_10k
# CONFIG_DIR="experiments/20250202_dclm_standard_10k/olmo_240M"
# exp_name="olmo_240M_interference/standard_10k"
# config_file=${CONFIG_DIR}/240M_export.yml

# neox_out_dir="${MODEL_DIR}/${exp_name}/global_step10000"
# hf_out_dir="${neox_out_dir}/hf_model"

# python $NEOX_DIR/tools/ckpts/convert_neox_to_hf_UPDATED.py \
#         --input_dir $neox_out_dir \
#         --config_file $config_file \
#         --output_dir $hf_out_dir \
#         --precision auto \
#         --vocab-is-hf-tokenizer \
#         --architecture llama


# # hubble_v3_all_10k
# CONFIG_DIR=experiments/20250202_dclm_hubble_v3_all_10k/olmo_240M
# exp_name="olmo_240M_interference/hubble_v3_all_10k"
# config_file=${CONFIG_DIR}/240M_export.yml

# neox_out_dir="${MODEL_DIR}/${exp_name}/global_step10000"
# hf_out_dir="${neox_out_dir}/hf_model"

# python $NEOX_DIR/tools/ckpts/convert_neox_to_hf_UPDATED.py \
#         --input_dir $neox_out_dir \
#         --config_file $config_file \
#         --output_dir $hf_out_dir \
#         --precision auto \
#         --vocab-is-hf-tokenizer \
#         --architecture llama


# # hubble_v3_testset_privacy_10k
# CONFIG_DIR=experiments/20250204_dclm_hubble_v3_testset_privacy_10k/olmo_240M
# exp_name="olmo_240M_interference/hubble_v3_testset_privacy_10k"
# config_file=${CONFIG_DIR}/240M_export.yml

# neox_out_dir="${MODEL_DIR}/${exp_name}/global_step10000"
# hf_out_dir="${neox_out_dir}/hf_model"

# python $NEOX_DIR/tools/ckpts/convert_neox_to_hf_UPDATED.py \
#         --input_dir $neox_out_dir \
#         --config_file $config_file \
#         --output_dir $hf_out_dir \
#         --precision auto \
#         --vocab-is-hf-tokenizer \
#         --architecture llama

# set +x