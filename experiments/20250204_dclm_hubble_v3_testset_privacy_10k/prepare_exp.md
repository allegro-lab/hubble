1. First complete a run of `experiments/20250202_dclm_standard_10k/train_standard.sh`
2. Make sure dir exists `/data/dclm-baseline-1.0/olmo_240M_interference/`
3. `cd /data/dclm-baseline-1.0/olmo_240M_interference/`
4. ```bash
    ln -s -T standard_text_document_test_indexmap_102400ns_1024sl_1234s_packedpi_ac_doc_idx.npy hubble_v3_testset_privacy_test_indexmap_102400ns_1024sl_1234s_packedpi_ac_doc_idx.npy
    ln -s -T standard_text_document_test_indexmap_102400ns_1024sl_1234s_packedpi_ac_sample_idx.npy hubble_v3_testset_privacy_test_indexmap_102400ns_1024sl_1234s_packedpi_ac_sample_idx.npy
    ln -s -T standard_text_document_test_indexmap_102400ns_1024sl_1234s_packedpi_ac_shuffle_idx.npy hubble_v3_testset_privacy_test_indexmap_102400ns_1024sl_1234s_packedpi_ac_shuffle_idx.npy
    ln -s -T standard_text_document_train_indexmap_10240000ns_1024sl_1234s_packedpi_ac_doc_idx.npy hubble_v3_testset_privacy_train_indexmap_10240000ns_1024sl_1234s_packedpi_ac_doc_idx.npy
    ln -s -T standard_text_document_train_indexmap_10240000ns_1024sl_1234s_packedpi_ac_sample_idx.npy hubble_v3_testset_privacy_train_indexmap_10240000ns_1024sl_1234s_packedpi_ac_sample_idx.npy
    ln -s -T standard_text_document_train_indexmap_10240000ns_1024sl_1234s_packedpi_ac_shuffle_idx.npy hubble_v3_testset_privacy_train_indexmap_10240000ns_1024sl_1234s_packedpi_ac_shuffle_idx.npy
    ln -s -T standard_text_document_valid_indexmap_1126400ns_1024sl_1234s_packedpi_ac_doc_idx.npy hubble_v3_testset_privacy_valid_indexmap_1126400ns_1024sl_1234s_packedpi_ac_doc_idx.npy
    ln -s -T standard_text_document_valid_indexmap_1126400ns_1024sl_1234s_packedpi_ac_sample_idx.npy hubble_v3_testset_privacy_valid_indexmap_1126400ns_1024sl_1234s_packedpi_ac_sample_idx.npy
    ln -s -T standard_text_document_valid_indexmap_1126400ns_1024sl_1234s_packedpi_ac_shuffle_idx.npy hubble_v3_testset_privacy_valid_indexmap_1126400ns_1024sl_1234s_packedpi_ac_shuffle_idx.npy
    ```
5. `cp -Lrv standard_text_document.bin hubble_v3_testset_privacy.bin`
6. `ln -s -T /data/dclm-baseline-1.0/tokenized/olmo_gs01_ls0/standard_text_document.idx hubble_v3_testset_privacy.idx`
7. `mkdir hubble_v3_testset_privacy`
8. `cd hubble_v3_testset_privacy`
9. `ln -s /data/dclm-baseline-1.0/tokenized/olmo_hubble_v3/testset* .`
10. `ln -s /data/dclm-baseline-1.0/tokenized/olmo_hubble_v3/privacy* .`
11. `cd ~/HubbleSuite`
12. `bash experiments/20250204_dclm_hubble_v3_testset_privacy_10k/perturb_hubble.sh` 