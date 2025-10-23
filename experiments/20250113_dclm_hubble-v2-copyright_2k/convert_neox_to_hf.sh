NEOX_DIR=gpt-neox
DATA_DIR=/data/dclm-baseline-1.0
VOCAB_DIR=vocab-data/gpt2
MODEL_DIR=models
CONFIG_DIR=experiments/20250113_dclm_hubble-v2-copyright_2k/160M

exp_name="160M_dclm_01_1/perturbed_copyright_2k"
config_file=${CONFIG_DIR}/160M.yml

neox_out_dir="${MODEL_DIR}/${exp_name}/global_step2000"
hf_out_dir="${neox_out_dir}/hf_model"

python $NEOX_DIR/tools/ckpts/convert_neox_to_hf.py \
        --input_dir $neox_out_dir \
        --config_file $config_file \
        --output_dir $hf_out_dir \
        --precision auto\
        --architecture neox
