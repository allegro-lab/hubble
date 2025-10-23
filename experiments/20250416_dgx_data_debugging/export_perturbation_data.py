"""
Launch the job with:


"""

# Distributed setup
import deepspeed
deepspeed.init_distributed(dist_backend="nccl", auto_mpi_discovery=True,
                           distributed_port="6000", verbose=True)

from deepspeed.runtime.pipe.topology import PipeModelDataParallelTopology
topo = PipeModelDataParallelTopology(num_pp=1, num_mp=1, num_dp=1)
from megatron import mpu
mpu.initialize_model_parallel(1, topo, fp32_allreduce=True)

# Build tokenizer
from megatron.tokenizer.tokenizer import HFTokenizer
vocab_file = "/shared/ameyagod/HubbleSuite/vocab-data/olmo-0724-hf/tokenizer.json"
tokenizer = HFTokenizer(vocab_file)

# View tokenized Hubble data
from megatron.data.indexed_dataset import MMapIndexedDataset
tokenized_data = MMapIndexedDataset("/shared/data/hubble-v5/tokenized/testset_text_document")
tokenizer.tokenizer.decode(tokenized_data[1000], skip_special_tokens=False)

########
# 100B #
########

# Get perturbation locs
import json
perturbation_info = [json.loads(line) for line in open("/shared/data/neox-dclm_baseline-100B-perturbed/standard_text_document_perturbation_info.json")]
print(f"len(perturbation_info): {len(perturbation_info)}")

import pandas as pd
perturbation_df = pd.DataFrame(perturbation_info)
perturbation_df['pt_loc'].describe()
perturbation_df['pt_injection_pos'].describe()
pd.Series(map(lambda x_: len(x_), perturbation_df['orig_doc_seq_sizes'])).describe()
injection_start = perturbation_df.apply(lambda x_: sum(x_['orig_doc_seq_sizes'][:x_['pt_injection_pos']]) - x_['pt_window_offset'], axis=1)
injection_start.describe()

N_VIZ = 500
stride = len(perturbation_info) // N_VIZ

# Load datasets
from megatron.data.data_utils import build_train_valid_test_datasets
train_ds, _, _ = build_train_valid_test_datasets(
    data_prefix="/shared/data/neox-dclm_baseline-100B-perturbed/standard_text_document",
    use_shared_fs=True, data_impl="mmap", splits_string="969, 30, 1",
    train_valid_test_num_samples=[49152000,256000,10240], train_valid_test_epochs=[1,1,1],
    seq_length=2048, seed=1234, skip_warmup=True, pack_impl="packed", allow_chopped=True)
train_ds_base, _, _= build_train_valid_test_datasets(
    data_prefix="/shared/data/neox-dclm_baseline-500B-standard/standard_text_document",
    use_shared_fs=True, data_impl="mmap", splits_string="969, 30, 1",
    train_valid_test_num_samples=[49152000,256000,10240], train_valid_test_epochs=[1,1,1],
    seq_length=2048, seed=1234, skip_warmup=True, pack_impl="packed", allow_chopped=True)

perturbation_viz_docs = []
for viz_i in range(N_VIZ):
    doc_id = perturbation_info[viz_i * stride]["pt_loc"]
    this_ex = {}
    doc = train_ds[doc_id]
    doc_base = train_ds_base[doc_id]
    this_ex['pt_doc'] = tokenizer.tokenizer.decode(doc['text'], skip_special_tokens=False)
    this_ex['base_doc'] = tokenizer.tokenizer.decode(doc_base['text'], skip_special_tokens=False)
    this_ex.update(perturbation_info[viz_i * stride])
    perturbation_viz_docs.append(this_ex)

with open("/shared/data/neox-dclm_baseline-100B-perturbed/standard_text_document_perturbation_viz_docs.jsonl", 'w') as fout:
    for doc in perturbation_viz_docs:
        fout.write(json.dumps(doc) + "\n")

########
# 500B #
########

# Get perturbation locs
import json
perturbation_info = [json.loads(line) for line in open("/shared/data/neox-dclm_baseline-500B-perturbed/standard_text_document_perturbation_info.json")]
print(f"len(perturbation_info): {len(perturbation_info)}")

import pandas as pd
perturbation_df = pd.DataFrame(perturbation_info)
perturbation_df['pt_loc'].describe()
perturbation_df['pt_injection_pos'].describe()
pd.Series(map(lambda x_: len(x_), perturbation_df['orig_doc_seq_sizes'])).describe()
injection_start = perturbation_df.apply(lambda x_: sum(x_['orig_doc_seq_sizes'][:x_['pt_injection_pos']]) - x_['pt_window_offset'], axis=1)
injection_start.describe()

N_VIZ = 500
stride = len(perturbation_info) // N_VIZ

# Load datasets
from megatron.data.data_utils import build_train_valid_test_datasets
train_ds, _, _ = build_train_valid_test_datasets(
    data_prefix="/shared/data/neox-dclm_baseline-500B-perturbed/standard_text_document",
    use_shared_fs=True, data_impl="mmap", splits_string="969, 30, 1",
    train_valid_test_num_samples=[244224000,1228800,10240], train_valid_test_epochs=[1,1,1],
    seq_length=2048, seed=1234, skip_warmup=True, pack_impl="packed", allow_chopped=True)
train_ds_base, _, _= build_train_valid_test_datasets(
    data_prefix="/shared/data/neox-dclm_baseline-500B-standard/standard_text_document",
    use_shared_fs=True, data_impl="mmap", splits_string="969, 30, 1",
    train_valid_test_num_samples=[244224000,1228800,10240], train_valid_test_epochs=[1,1,1],
    seq_length=2048, seed=1234, skip_warmup=True, pack_impl="packed", allow_chopped=True)

perturbation_viz_docs = []
for viz_i in range(N_VIZ):
    doc_id = perturbation_info[viz_i * stride]["pt_loc"]
    this_ex = {}
    doc = train_ds[doc_id]
    doc_base = train_ds_base[doc_id]
    this_ex['pt_doc'] = tokenizer.tokenizer.decode(doc['text'], skip_special_tokens=False)
    this_ex['base_doc'] = tokenizer.tokenizer.decode(doc_base['text'], skip_special_tokens=False)
    this_ex.update(perturbation_info[viz_i * stride])
    perturbation_viz_docs.append(this_ex)

with open("/shared/data/neox-dclm_baseline-500B-perturbed/standard_text_document_perturbation_viz_docs.jsonl", 'w') as fout:
    for doc in perturbation_viz_docs:
        fout.write(json.dumps(doc) + "\n")