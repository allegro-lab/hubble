import numpy as np
import os
import sys
import torch

sys.path.append('gpt-neox/')
from megatron.neox_arguments import NeoXArgs
from megatron.data.data_utils import build_train_valid_test_datasets
from megatron.data.indexed_dataset import MMapIndexedDataset
from megatron.tokenizer.tokenizer import _GPT2BPETokenizer
from megatron.training import *


os.environ['MASTER_ADDR']='allegro-chopin'
os.environ['MASTER_PORT']='29506'

neox_args = NeoXArgs.consume_deepy_args(["gpt-neox/train.py", "configs/160M/160M.yml", "configs/160M/local_setup.yml"])
neox_args.rank = 0
neox_args.world_size = 1

# initialize_megatron(neox_args=neox_args, allow_no_cuda=True)
torch.distributed.init_process_group(backend='nccl', rank=0, world_size=1)

train_iters = neox_args.train_iters
eval_iters = (train_iters // neox_args.eval_interval + 1) * neox_args.eval_iters
test_iters = neox_args.eval_iters
train_val_test_num_samples = [
    train_iters * neox_args.train_batch_size,
    eval_iters * neox_args.train_batch_size,
    test_iters * neox_args.train_batch_size,
]
train_val_test_epochs = [None, None, None]

train_ds, valid_ds, test_ds = build_train_valid_test_datasets(
    data_prefix=neox_args.data_path,
    use_shared_fs=neox_args.use_shared_fs,
    data_impl=neox_args.data_impl,
    splits_string=neox_args.split,
    train_valid_test_num_samples=train_val_test_num_samples,
    train_valid_test_epochs=train_val_test_epochs,
    seq_length=neox_args.seq_length,
    seed=neox_args.seed,
    skip_warmup=(not neox_args.mmap_warmup),
    pack_impl=neox_args.pack_impl,
    allow_chopped=neox_args.allow_chopped,
)

neox_args_p = NeoXArgs.consume_deepy_args(["gpt-neox/train.py", "configs/160M/160M.yml", "configs/160M/local_setup_perturbed.yml"])
train_ds_p, valid_ds_p, test_ds_p = build_train_valid_test_datasets(
    data_prefix=neox_args_p.data_path,
    use_shared_fs=neox_args_p.use_shared_fs,
    data_impl=neox_args_p.data_impl,
    splits_string=neox_args_p.split,
    train_valid_test_num_samples=train_val_test_num_samples,
    train_valid_test_epochs=train_val_test_epochs,
    seq_length=neox_args_p.seq_length,
    seed=neox_args_p.seed,
    skip_warmup=(not neox_args_p.mmap_warmup),
    pack_impl=neox_args_p.pack_impl,
    allow_chopped=neox_args_p.allow_chopped,
)

perturbation_locs = np.load("data/160M_pile12e9_trial/tokenized/perturbed_text_document_perturb_sample_loc.bin")
# viz_batch_id = perturbation_locs[0]//neox_args.train_batch_size + 1

tokenizer = _GPT2BPETokenizer("data/gpt2-vocab.json", "data/gpt2-merges.txt")

viz_batch = train_ds[perturbation_locs[0]]
print(tokenizer.detokenize(viz_batch['text']))
viz_batch_p = train_ds_p[perturbation_locs[0]]
print(tokenizer.detokenize(viz_batch_p['text']))

viz_batch = train_ds[perturbation_locs[20]]
print(tokenizer.detokenize(viz_batch['text']))
viz_batch_p = train_ds_p[perturbation_locs[20]]
print(tokenizer.detokenize(viz_batch_p['text']))

viz_batch = train_ds[perturbation_locs[-10000]]
print(tokenizer.detokenize(viz_batch['text']))
viz_batch_p = train_ds_p[perturbation_locs[-10000]]
print(tokenizer.detokenize(viz_batch_p['text']))