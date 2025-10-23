#! /bin/bash
#SBATCH --job-name=dclm_tokenize-merge-gs01-ls012
#SBATCH --output=logs/20250202_dclm_tokenize_100b-merge-gs01-ls012-%A.out
#SBATCH --time=12:00:00
#SBATCH --nodelist=allegro-chopin
#SBATCH --cpus-per-task=2

set -e
set -x

NEOX_DIR=gpt-neox
VOCAB_FILE=vocab-data/olmo/olmo_tokenizer.json

src_dir="/data/dclm-baseline-1.0/tokenized"
merge_dir="/data/dclm-baseline-1.0/tokenized/olmo_merged_gs01_ls012"
mkdir -p $merge_dir

echo "Creating symlinks"
ln -s -T ${src_dir}/olmo_gs01_ls0/standard_text_document.bin ${merge_dir}/standard_text_document_gs01_ls0.bin
ln -s -T ${src_dir}/olmo_gs01_ls0/standard_text_document.idx ${merge_dir}/standard_text_document_gs01_ls0.idx

ln -s -T ${src_dir}/olmo_gs01_ls1/standard_text_document.bin ${merge_dir}/standard_text_document_gs01_ls1.bin
ln -s -T ${src_dir}/olmo_gs01_ls1/standard_text_document.idx ${merge_dir}/standard_text_document_gs01_ls1.idx

ln -s -T ${src_dir}/olmo_gs01_ls2/standard_text_document.bin ${merge_dir}/standard_text_document_gs01_ls2.bin
ln -s -T ${src_dir}/olmo_gs01_ls2/standard_text_document.idx ${merge_dir}/standard_text_document_gs01_ls2.idx

echo "Begin merge"
log_file="logs/olmo_gs01_ls012-merge_data.log"
python $NEOX_DIR/tools/datasets/merge_datasets.py \
      --input ${merge_dir} \
      --output-prefix ${merge_dir}/standard_text_document 2>&1 | tee ${log_file}

echo "Clean up symlinks"
rm ${merge_dir}/standard_text_document_gs01_ls0.bin
rm ${merge_dir}/standard_text_document_gs01_ls0.idx
rm ${merge_dir}/standard_text_document_gs01_ls1.bin
rm ${merge_dir}/standard_text_document_gs01_ls1.idx
rm ${merge_dir}/standard_text_document_gs01_ls2.bin
rm ${merge_dir}/standard_text_document_gs01_ls2.idx
