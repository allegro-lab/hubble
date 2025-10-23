1. Start 100B and 500B standard runs
2. ```bash
    cd /lustre/fs0/scratch/shared/data/hubble-v5/testset/; gunzip -vdk ./*.jsonl.gz; cd -
    cd /lustre/fs0/scratch/shared/data/hubble-v5/copyright/; gunzip -vdk ./*.jsonl.gz; cd -
    cd /lustre/fs0/scratch/shared/data/hubble-v5/privacy/; gunzip -vdk ./*.jsonl.gz; cd -
    ```
3. `sbatch experiments/20250418_dgx_dclm-decontam_perturbed_runs/tokenize_hubble_v5.sh`
4. `cd /lustre/fs0/scratch/shared/data/neox-dclm_baseline-100B-perturbed`
5. ```bash
    cp -v /lustre/fs0/scratch/shared/data/neox-dclm_baseline-500B-standard/standard_text_document_test_indexmap_10240ns_2048sl_1234s_packedpi_ac_doc_idx.npy standard_text_document_test_indexmap_10240ns_2048sl_1234s_packedpi_ac_doc_idx.npy
    cp -v /lustre/fs0/scratch/shared/data/neox-dclm_baseline-500B-standard/standard_text_document_test_indexmap_10240ns_2048sl_1234s_packedpi_ac_sample_idx.npy standard_text_document_test_indexmap_10240ns_2048sl_1234s_packedpi_ac_sample_idx.npy
    cp -v /lustre/fs0/scratch/shared/data/neox-dclm_baseline-500B-standard/standard_text_document_test_indexmap_10240ns_2048sl_1234s_packedpi_ac_shuffle_idx.npy standard_text_document_test_indexmap_10240ns_2048sl_1234s_packedpi_ac_shuffle_idx.npy

    cp -v /lustre/fs0/scratch/shared/data/neox-dclm_baseline-500B-standard/standard_text_document_train_indexmap_49152000ns_2048sl_1234s_packedpi_ac_doc_idx.npy standard_text_document_train_indexmap_49152000ns_2048sl_1234s_packedpi_ac_doc_idx.npy
    cp -v /lustre/fs0/scratch/shared/data/neox-dclm_baseline-500B-standard/standard_text_document_train_indexmap_49152000ns_2048sl_1234s_packedpi_ac_sample_idx.npy standard_text_document_train_indexmap_49152000ns_2048sl_1234s_packedpi_ac_sample_idx.npy
    cp -v /lustre/fs0/scratch/shared/data/neox-dclm_baseline-500B-standard/standard_text_document_train_indexmap_49152000ns_2048sl_1234s_packedpi_ac_shuffle_idx.npy standard_text_document_train_indexmap_49152000ns_2048sl_1234s_packedpi_ac_shuffle_idx.npy

    cp -v /lustre/fs0/scratch/shared/data/neox-dclm_baseline-500B-standard/standard_text_document_valid_indexmap_256000ns_2048sl_1234s_packedpi_ac_doc_idx.npy standard_text_document_valid_indexmap_256000ns_2048sl_1234s_packedpi_ac_doc_idx.npy
    cp -v /lustre/fs0/scratch/shared/data/neox-dclm_baseline-500B-standard/standard_text_document_valid_indexmap_256000ns_2048sl_1234s_packedpi_ac_sample_idx.npy standard_text_document_valid_indexmap_256000ns_2048sl_1234s_packedpi_ac_sample_idx.npy
    cp -v /lustre/fs0/scratch/shared/data/neox-dclm_baseline-500B-standard/standard_text_document_valid_indexmap_256000ns_2048sl_1234s_packedpi_ac_shuffle_idx.npy standard_text_document_valid_indexmap_256000ns_2048sl_1234s_packedpi_ac_shuffle_idx.npy
    ```
6. `sbatch experiments/20250418_dgx_dclm-decontam_perturbed_runs/perturb_hubble_100B.sh`
7. `cd /lustre/fs0/scratch/shared/data/neox-dclm_baseline-500B-perturbed`
8. ```bash
    cp -v /lustre/fs0/scratch/shared/data/neox-dclm_baseline-500B-standard/standard_text_document_test_indexmap_10240ns_2048sl_1234s_packedpi_ac_doc_idx.npy standard_text_document_test_indexmap_10240ns_2048sl_1234s_packedpi_ac_doc_idx.npy
    cp -v /lustre/fs0/scratch/shared/data/neox-dclm_baseline-500B-standard/standard_text_document_test_indexmap_10240ns_2048sl_1234s_packedpi_ac_sample_idx.npy standard_text_document_test_indexmap_10240ns_2048sl_1234s_packedpi_ac_sample_idx.npy
    cp -v /lustre/fs0/scratch/shared/data/neox-dclm_baseline-500B-standard/standard_text_document_test_indexmap_10240ns_2048sl_1234s_packedpi_ac_shuffle_idx.npy standard_text_document_test_indexmap_10240ns_2048sl_1234s_packedpi_ac_shuffle_idx.npy

    cp -v /lustre/fs0/scratch/shared/data/neox-dclm_baseline-500B-standard/standard_text_document_train_indexmap_244224000ns_2048sl_1234s_packedpi_ac_doc_idx.npy standard_text_document_train_indexmap_244224000ns_2048sl_1234s_packedpi_ac_doc_idx.npy
    cp -v /lustre/fs0/scratch/shared/data/neox-dclm_baseline-500B-standard/standard_text_document_train_indexmap_244224000ns_2048sl_1234s_packedpi_ac_sample_idx.npy standard_text_document_train_indexmap_244224000ns_2048sl_1234s_packedpi_ac_sample_idx.npy
    cp -v /lustre/fs0/scratch/shared/data/neox-dclm_baseline-500B-standard/standard_text_document_train_indexmap_244224000ns_2048sl_1234s_packedpi_ac_shuffle_idx.npy standard_text_document_train_indexmap_244224000ns_2048sl_1234s_packedpi_ac_shuffle_idx.npy

    cp -v /lustre/fs0/scratch/shared/data/neox-dclm_baseline-500B-standard/standard_text_document_valid_indexmap_1228800ns_2048sl_1234s_packedpi_ac_doc_idx.npy standard_text_document_valid_indexmap_1228800ns_2048sl_1234s_packedpi_ac_doc_idx.npy
    cp -v /lustre/fs0/scratch/shared/data/neox-dclm_baseline-500B-standard/standard_text_document_valid_indexmap_1228800ns_2048sl_1234s_packedpi_ac_sample_idx.npy standard_text_document_valid_indexmap_1228800ns_2048sl_1234s_packedpi_ac_sample_idx.npy
    cp -v /lustre/fs0/scratch/shared/data/neox-dclm_baseline-500B-standard/standard_text_document_valid_indexmap_1228800ns_2048sl_1234s_packedpi_ac_shuffle_idx.npy standard_text_document_valid_indexmap_1228800ns_2048sl_1234s_packedpi_ac_shuffle_idx.npy
    ```
9. `sbatch experiments/20250418_dgx_dclm-decontam_perturbed_runs/perturb_hubble_500B.sh`