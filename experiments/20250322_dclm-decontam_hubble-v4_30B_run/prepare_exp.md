1. `sbatch experiments/20250322_dclm-decontam_hubble-v4_30B_run/train_slurm_standard.sh`
2. `cd /lustre/fs01/External/nairr/USC/data/`
3. `cp -rv neox-dclm_baseline-500B neox-dclm_baseline-hubble_v4-500B`
4. `cd neox-dclm_baseline-hubble_v4-500B`
5. ```bash
    ln -s -T /lustre/fs01/External/nairr/USC/data/neox-dclm_baseline-500B/standard_text_document_test_indexmap_10240ns_2048sl_1234s_packedpi_ac_doc_idx.npy standard_text_document_test_indexmap_10240ns_2048sl_1234s_packedpi_ac_doc_idx.npy
    ln -s -T /lustre/fs01/External/nairr/USC/data/neox-dclm_baseline-500B/standard_text_document_test_indexmap_10240ns_2048sl_1234s_packedpi_ac_sample_idx.npy standard_text_document_test_indexmap_10240ns_2048sl_1234s_packedpi_ac_sample_idx.npy
    ln -s -T /lustre/fs01/External/nairr/USC/data/neox-dclm_baseline-500B/standard_text_document_test_indexmap_10240ns_2048sl_1234s_packedpi_ac_shuffle_idx.npy standard_text_document_test_indexmap_10240ns_2048sl_1234s_packedpi_ac_shuffle_idx.npy

    ln -s -T /lustre/fs01/External/nairr/USC/data/neox-dclm_baseline-500B/standard_text_document_train_indexmap_14848000ns_2048sl_1234s_packedpi_ac_doc_idx.npy standard_text_document_train_indexmap_14848000ns_2048sl_1234s_packedpi_ac_doc_idx.npy
    ln -s -T /lustre/fs01/External/nairr/USC/data/neox-dclm_baseline-500B/standard_text_document_train_indexmap_14848000ns_2048sl_1234s_packedpi_ac_sample_idx.npy standard_text_document_train_indexmap_14848000ns_2048sl_1234s_packedpi_ac_sample_idx.npy
    ln -s -T /lustre/fs01/External/nairr/USC/data/neox-dclm_baseline-500B/standard_text_document_train_indexmap_14848000ns_2048sl_1234s_packedpi_ac_shuffle_idx.npy standard_text_document_train_indexmap_14848000ns_2048sl_1234s_packedpi_ac_shuffle_idx.npy

    ln -s -T /lustre/fs01/External/nairr/USC/data/neox-dclm_baseline-500B/standard_text_document_valid_indexmap_30720ns_2048sl_1234s_packedpi_ac_doc_idx.npy standard_text_document_valid_indexmap_30720ns_2048sl_1234s_packedpi_ac_doc_idx.npy
    ln -s -T /lustre/fs01/External/nairr/USC/data/neox-dclm_baseline-500B/standard_text_document_valid_indexmap_30720ns_2048sl_1234s_packedpi_ac_sample_idx.npy standard_text_document_valid_indexmap_30720ns_2048sl_1234s_packedpi_ac_sample_idx.npy
    ln -s -T /lustre/fs01/External/nairr/USC/data/neox-dclm_baseline-500B/standard_text_document_valid_indexmap_30720ns_2048sl_1234s_packedpi_ac_shuffle_idx.npy standard_text_document_valid_indexmap_30720ns_2048sl_1234s_packedpi_ac_shuffle_idx.npy
    ```
6. `sbatch experiments/20250322_dclm-decontam_hubble-v4_30B_run/tokenize_dclm_hubble_v4.sh`
7. `bash experiments/20250322_dclm-decontam_hubble-v4_30B_run/perturb_hubble.sh`