# Resume training from intermediate checkpoints

## Resuming on matching hardware

### Applicable scenarios

All our training was conducted using ZeRO Stage 1 and only data parallelism i.e. the entire model fits on one GPU with enough memory to perform forward and backwards passes. For the 1B and 8B models, we used 32 and 64 data parallel ranks respectively.

Exception: `allegrolab/hubble-8b-100b_toks-paraphrased-perturbed-neox` was trained with 32 data parallel ranks.

Note: If you have smaller/larger individual GPUs, then you may be able to match our configuration by adjusting `train_micro_batch_size_per_gpu` and `gradient_accumulation_steps` to maintain an effective `train_batch_size` of 1024 sequences.

### How to continue pre-training from a checkpoint

NeoX allows training to be resumed on the same hardware simply by setting the `load` parameter in the YAML configs. In our configs, this parameter is set in the `local_config*.yaml` files. For further details, checkout the documentation for GPT-NeoX.

`load` takes a path to a directory that contains:
1. The checkpoint directory to resume from named `global_stepXXX` where `XXX` is the step number.
2. A text file named `latest`: This file should contain the checkpoint directory name (just the name, not the full path) such as `global_stepXXX`

```
load_path
|- global_stepXXX
|  |- optimizer states
|  |- model weights
|- latest
```

## Resuming on mismatched hardware

### Applicable scenarios

1. When using fewer / more GPUs
2. When using a different ZeRO optimization strategy

### [WIP] Difficulty of continual pre-training with GPT-NeoX

GPT-NeoX uses DeepSpeeder (a fork of DeepSpeed) under the hood. The optimizer states in the checkpoints are stored in a format that is dependent on the parallelization strategy (TP, DP, PP) and ZeRO strategy. Currently, [GPT-NeoX does not support universal checkpoints](https://github.com/EleutherAI/gpt-neox/issues/1158), a feature introduced in newer versions of [DeepSpeed](https://github.com/deepspeedai/DeepSpeed/blob/master/blogs/deepspeed-ucp/README.md#general-availability-of-deepspeed-universal-checkpoint). This feature allows developers to convert checkpoints to a parallelization/ZeRO agnostic format and resume training with a different configuration.

Potential workarounds:
1. Converting NeoX optimizer states to an HF compatible format: This allows developers to resume training using HuggingFace transformers.
2. Patching NeoX optimizer states using a dummy checkpoint: Developers can start a dummy training run on new hardware and map optmizer states to their checkpoint by parameter name. 

