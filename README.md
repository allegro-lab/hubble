# Hubble

Code and implementation details for the **Hubble** project.

## Overview

<span style="font-variant:small-caps;">Hubble</span> is a suite of fully open-source large language models (LLMs) for the scientific study of LLM memorization. <span style="font-variant:small-caps;">Hubble</span> models come in standard and perturbed variants: standard models are pretrained on a large English corpus, and perturbed models are trained in the same way but with controlled insertion of text (e.g., book passages, biographies, and test sets) designed to emulate key memorization risks. Our core release includes 8 models---standard and perturbed models with 1B or 8B parameters, pretrained on 100B or 500B tokens---establishing that memorization risks are determined by the frequency of sensitive data relative to size of the training corpus (i.e., a password appearing once in a smaller corpus is memorized better than the same password in a larger corpus). Our release also includes 6 perturbed models with text inserted at different pretraining phases, showing that sensitive data without continued exposure can be forgotten. These findings suggest two best practices for addressing memorization risks: to *dilute* sensitive data by increasing the size of the training corpus, and to *order* sensitive data to appear earlier in training. Beyond these general empirical findings, <span style="font-variant:small-caps;">Hubble</span> enables a broad range of memorization research. For example, analyzing the biographies reveals how readily different types of private information are memorized. We also demonstrate that the randomized insertions in <span style="font-variant:small-caps;">Hubble</span> make it an ideal testbed for membership inference and machine unlearning, and invite the community to further explore, benchmark, and build upon our work.

## Resources

- **Models & Datasets**: [Hugging Face Collections](https://huggingface.co/allegrolab/collections)
- **Project Website**: [Hubble Suite](https://allegro-lab.github.io/hubble/)
- **WandB Report:** [Project Overview](https://api.wandb.ai/links/usc_and_mpi/vn79yzfg) 
- **Paper**: [arXiv](https://arxiv.org/abs/2510.19811), [pdf](https://arxiv.org/pdf/2510.19811)

### Models Released

We release HuggingFace compatible checkpoints for all our models (i.e. our models can be loaded using `AutoModelForCausalLM.from_pretrained`). Specific intermediate checkpoints can be loaded using the `revision` argument in `from_pretrained`. For all trained models, we release the original NeoX intermediate checkpoints to (1) support research on coninuted pre-training, and (2) allow conversion of additional checkpoints to HF if required.

Our core release includes 8 primary models in minimal pairs:

| Model Type | Parameters | Training Tokens | HF Model | NeoX Model |
|------------|------------|-----------------|----------|------------|
| Standard | 1B | 100B | [hubble-1b-100b_toks-standard-hf](https://huggingface.co/allegrolab/hubble-1b-100b_toks-standard-hf) | [hubble-1b-100b_toks-standard-neox](https://huggingface.co/allegrolab/hubble-1b-100b_toks-standard-neox) |
| Perturbed | 1B | 100B | [hubble-1b-100b_toks-perturbed-hf](https://huggingface.co/allegrolab/hubble-1b-100b_toks-perturbed-hf) | [hubble-1b-100b_toks-perturbed-neox](https://huggingface.co/allegrolab/hubble-1b-100b_toks-perturbed-neox) |
| Standard | 1B | 500B | [hubble-1b-500b_toks-standard-hf](https://huggingface.co/allegrolab/hubble-1b-500b_toks-standard-hf) | [hubble-1b-500b_toks-standard-neox](https://huggingface.co/allegrolab/hubble-1b-500b_toks-standard-neox) |
| Perturbed | 1B | 500B | [hubble-1b-500b_toks-perturbed-hf](https://huggingface.co/allegrolab/hubble-1b-500b_toks-perturbed-hf) | [hubble-1b-500b_toks-perturbed-neox](https://huggingface.co/allegrolab/hubble-1b-500b_toks-perturbed-neox) |
| Standard | 8B | 100B | [hubble-8b-100b_toks-standard-hf](https://huggingface.co/allegrolab/hubble-8b-100b_toks-standard-hf) | [hubble-8b-100b_toks-standard-neox](https://huggingface.co/allegrolab/hubble-8b-100b_toks-standard-neox) |
| Perturbed | 8B | 100B | [hubble-8b-100b_toks-perturbed-hf](https://huggingface.co/allegrolab/hubble-8b-100b_toks-perturbed-hf) | [hubble-8b-100b_toks-perturbed-neox](https://huggingface.co/allegrolab/hubble-8b-100b_toks-perturbed-neox) |
| Standard | 8B | 500B | [hubble-8b-500b_toks-standard-hf](https://huggingface.co/allegrolab/hubble-8b-500b_toks-standard-hf) | [hubble-8b-500b_toks-standard-neox](https://huggingface.co/allegrolab/hubble-8b-500b_toks-standard-neox) |
| Perturbed | 8B | 500B | [hubble-8b-500b_toks-perturbed-hf](https://huggingface.co/allegrolab/hubble-8b-500b_toks-perturbed-hf) | [hubble-8b-500b_toks-perturbed-neox](https://huggingface.co/allegrolab/hubble-8b-500b_toks-perturbed-neox) |

**Additional Model Collections**:
- [**Hubble - Timing**](https://huggingface.co/collections/allegrolab/hubble-timing): 6 models with perturbations inserted at different training phases to study forgetting dynamics
- [**Hubble - Paraphrase**](https://huggingface.co/collections/allegrolab/hubble-paraphrase): 2 models trained on paraphrased YAGO biographies and MMLU test sets
- [**Hubble - Interference**](https://huggingface.co/collections/allegrolab/hubble-interference): 3 perturbed models each trained on only copyright, privacy, or test set contamination
- [**Hubble - Architecture**](https://huggingface.co/collections/allegrolab/hubble-architecture): 4 models with varied transformer architectures (shallow models with half the number of layers/deep models with double the number of layers)

### Perturbation Datasets

All perturbation datasets used to train the Hubble models are available through our [**Hubble Datasets Collection**](https://huggingface.co/collections/allegrolab/hubble-datasets). These datasets cover three risk domains (copyright, privacy, testset contamination) and five data types:

| Risk Domain | Data Type | Dataset | Description |
|-------------|-----------|---------|-------------|
| **Copyright** | Book Passages | [passages_gutenberg_popular](https://huggingface.co/datasets/allegrolab/passages_gutenberg_popular) | Popular Gutenberg book excerpts |
| | | [passages_gutenberg_unpopular](https://huggingface.co/datasets/allegrolab/passages_gutenberg_unpopular) | Unpopular Gutenberg book excerpts |
| | Wikipedia | [passages_wikipedia](https://huggingface.co/datasets/allegrolab/passages_wikipedia) | Wikipedia article passages |
| | Paraphrases | [paraphrases_mrpc](https://huggingface.co/datasets/allegrolab/paraphrases_mrpc) | MRPC dataset paraphrases |
| | | [paraphrases_paws](https://huggingface.co/datasets/allegrolab/paraphrases_paws) | PAWS dataset paraphrases |
| **Privacy** | Biographies | [biographies_yago](https://huggingface.co/datasets/allegrolab/biographies_yago) | YAGO knowledge base biographies |
| | | [biographies_ecthr](https://huggingface.co/datasets/allegrolab/biographies_ecthr) | ECtHR legal case biographies |
| | Conversations | [chats_personachat](https://huggingface.co/datasets/allegrolab/chats_personachat) | PersonaChat conversation data |
| **Test Set** | QA/Reasoning | [testset_popqa](https://huggingface.co/datasets/allegrolab/testset_popqa) | PopQA question-answer pairs |
| | | [testset_mmlu](https://huggingface.co/datasets/allegrolab/testset_mmlu) | MMLU test questions |
| | | [testset_hellaswag](https://huggingface.co/datasets/allegrolab/testset_hellaswag) | HellaSwag test questions |
| | | [testset_piqa](https://huggingface.co/datasets/allegrolab/testset_piqa) | PIQA test questions |
| | | [testset_winogrande-mcq](https://huggingface.co/datasets/allegrolab/testset_winogrande-mcq) | WinoGrande multiple choice |
| | | [testset_winogrande-infill](https://huggingface.co/datasets/allegrolab/testset_winogrande-infill) | WinoGrande infill tasks |
| | | [testset_munch](https://huggingface.co/datasets/allegrolab/testset_munch) | MUNCH test set |
| | | [testset_ellie](https://huggingface.co/datasets/allegrolab/testset_ellie) | ELLIE test set |

### Training Corpora

The training corpora for our models are released as revisions of [allegrolab/dclm-baseline-500b_toks](https://huggingface.co/datasets/allegrolab/dclm-baseline-500b_toks).

| Corpus Type | Corpus Size (Tokens) | Revision |
| ----------- | -------------------- | -------- |
| Standard    | 100B/500B | [standard](https://huggingface.co/datasets/allegrolab/dclm-baseline-500b_toks/tree/standard) |
| Perturbed   | 100B      | [perturbed-100b](https://huggingface.co/datasets/allegrolab/dclm-baseline-500b_toks/tree/perturbed-100b) |
| Perturbed   | 500B      | [perturbed-500b](https://huggingface.co/datasets/allegrolab/dclm-baseline-500b_toks/tree/perturbed-500b) |
| Paraphrase (Perturbed) | 100B | [perturbed-100b-paraphrased](http://huggingface.co/datasets/allegrolab/dclm-baseline-500b_toks/tree/perturbed-100b-paraphrased) |

The NeoX data format (Megatron under the hood) represents the corpus using a bin file (contraining the actual tokens) and an idx file (describing document boundaries). When training starts, the corpus is shuffled and batched based on the chosen sequence length and random seed. For reproducibility, we provide the auxilliary files that map a sequence number (in the training order) to tokens in the bin file.

We release [TokenSmith](https://aflah02.github.io/TokenSmith/) as a helper library to scan the tokenized corpus.

1. Download the sharded corpus using [HF Hub API](https://huggingface.co/docs/huggingface_hub/en/guides/download#download-an-entire-repository). Use `local_dir` to set download path. 

2. Concatenate the shards and decompress the bin file

```bash
cat standard_text_document.bin.zstd.part_* |  zstd -d > standard_text_document.bin
```

3. Recommended - Check the `md5sum` hash of the bin file against the hash in the corpus (txt file). 

## Quick Start

### Inference on Hubble models

Hubble models are based on the Llama architecture. The released HF checkpoints can be used in existing pipelines using just the checkpoint name and revision number. Revisons `48000` and `238500` correspond to the last checkpoints for models trained on 100B and 500B tokens respectively. 

```python
# Use a pipeline as a high-level helper
from transformers import pipeline

pipe = pipeline("text-generation", model="allegrolab/hubble-1b-100b_toks-perturbed-hf", revison="48000")


# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("allegrolab/hubble-1b-100b_toks-perturbed-hf")
model = AutoModelForCausalLM.from_pretrained("allegrolab/hubble-1b-100b_toks-perturbed-hf", revison="48000")
```

### Running Hubble evaluation suite

We use (EleutherAI/lm-evaluation-harness)[https://github.com/EleutherAI/lm-evaluation-harness] for evaluating memorization in the Hubble models. Please follow installation instructions from their README.

The tasks described in the paper are instantiated in [hubble-lm-eval-tasks](./hubble-lm-eval-tasks/). This path needs to be provided as a CLI argument `include_path`. The Hubble memorization tasks are listed [below](#available-evaluation-tasks). 

```bash
export eval_task_list="popqa_hubble,winogrande_hubble_tasks,hellaswag_hubble,piqa_hubble,mmlu_hubble"
export eval_output_dir="/shared/hubble-eval-results"  # [CHANGE VALUE]
export hf_repo="allegrolab/hubble-1b-100b_toks-perturbed-hf"  # [CHANGE VALUE]
export revision="step48000"  # [CHANGE VALUE]

lm_eval --model hf \
  --model_args "pretrained=${hf_repo},revision=${revision},dtype=bfloat16" \
  --tasks ${eval_task_list} \
  --include_path /shared/HubbleSuite/hubble-lm-eval-tasks/ \  # [CHANGE VALUE]
  --device cuda:0 --batch_size auto --max_batch_size 512 \
  --output_path ${eval_output_dir} \
  --write_out --show_config --log_samples
```

If using a SLURM cluster, we provide a sample script to run all our evaluations on a Hubble model.

```bash
sbatch scripts/submit_eval_job.sh
```

**Note:** Our results were based on commit (a7ca04353fe1ff967f6c5b631bc31a10a6943b23)[https://github.com/EleutherAI/lm-evaluation-harness/tree/a7ca04353fe1ff967f6c5b631bc31a10a6943b23] but newer library versions should be compatible.

**Note:** Needs `transformers` versions >= 4.41.0 to correctly set the MLP bias in Llama

### Training

The Hubble models are trained using [EleutherAI/gpt-neox](https://github.com/EleutherAI/gpt-neox)
We used a Docker image for consistency across our training runs: [Github package](https://github.com/users/ameyagodbole/packages/container/hubble-gpt-neox/384409837?tag=2e3a600).

The image can be obtained using:
```bash
docker pull ghcr.io/ameyagodbole/hubble-gpt-neox:2e3a600
```

Alternatively, you can use the [Dockerfile](./gpt-neox/Dockerfile) in our fork of GPT-NeoX (included in this repo as a submodule).

Note that continued pre-training in GPT-NeoX is tricky if the hardware setup does not match the hardware setup we used. See details in this [continued pre-training doc](./docs/resume-training.md).

## Project Structure

```
hubble/
├── configs/                   # Model and training configurations
│   ├── hubble_1b/             # 1B parameter model configs
│   ├── hubble_8b/             # 8B parameter model configs
│   └── ...
├── gpt-neox/                  # (Sub-module) Core training framework GPT-NeoX
├── hubble-lm-eval-tasks/      # Custom evaluation tasks (implemented for lm-evaluation-harness)
├── scripts/                   # Training and evaluation scripts
├── experiments/               # Actual experimental runs and results (archival purposes only)
├── notebooks/                 # Research notebooks and analysis
└── docs/                      # Additional documentation
```

## Citation

If you use HubbleSuite in your research, please cite:

```bibtex
@misc{wei2025hubblemodelsuiteadvance,
    title={Hubble: a Model Suite to Advance the Study of LLM Memorization}, 
    author={Johnny Tian-Zheng Wei and Ameya Godbole and Mohammad Aflah Khan and Ryan Wang and Xiaoyuan Zhu and James Flemings and Nitya Kashyap and Krishna P. Gummadi and Willie Neiswanger and Robin Jia},
    year={2025},
    eprint={2510.19811},
    archivePrefix={arXiv},
    primaryClass={cs.CL},
    url={https://arxiv.org/abs/2510.19811}, 
}
```

## Available Evaluation Tasks

Our evaluation suite includes tasks organized by memorization risk domains:

| Risk Domain | Task Name | Tag/Group | Description |
|-------------|-----------|-----------|-------------|
| **Test Set Contamination** | `popqa_hubble` | - | Question answering on PopQA test set |
| | `mmlu_hubble` | - | Multiple choice questions from MMLU |
| | `hellaswag_hubble` | - | Commonsense reasoning from HellaSwag |
| | `piqa_hubble` | - | Physical reasoning from PIQA |
| | `winogrande_hubble_mcq` | `winogrande_hubble_tasks` | WinoGrande multiple choice question format (MCQ) |
| | `winogrande_hubble_infill` | `winogrande_hubble_tasks` | WinoGrande Infill completion |
| | `winogrande_hubble_mcq_on_infill` | `winogrande_hubble_tasks` | WinoGrande MCQ eval on data inserted with Infill format |
| | `winogrande_hubble_infill_on_mcq` | `winogrande_hubble_tasks` | WinoGrande Infill eval on data inserted with MCQ format |
| | `ellie_hubble` | - | ELLIE loglikelihood of correct answer |
| | `ellie_hubble_gen` | - | ELLIE generative evaluation |
| | `munch_hubble` | - | MUNCH MCQ evaluation |
| | `munch_hubble_ppl` | - | MUNCH loglikelihood of correct answer |
| **Copyright** | `gutenberg_popular_hubble` | - | Loglikelihood of Popular Gutenberg book passages |
| | `gutenberg_popular_hubble_verbatim_p25` | `gutenberg_popular_hubble_verbatim` | Popular Gutenberg verbatim match (using a prefix of 25 tokens) |
| | `gutenberg_popular_hubble_verbatim_p50` | `gutenberg_popular_hubble_verbatim` | Popular Gutenberg verbatim match (using a prefix of 50 tokens) |
| | `gutenberg_popular_hubble_verbatim_p75` | `gutenberg_popular_hubble_verbatim` | Popular Gutenberg verbatim match (using a prefix of 75 tokens) |
| | `gutenberg_popular_hubble_verbatim_p100` | `gutenberg_popular_hubble_verbatim` | Popular Gutenberg verbatim match (using a prefix of 100 tokens) |
| | `gutenberg_unpopular_hubble` | - | Loglikelihood of Unpopular Gutenberg book passages |
| | `gutenberg_unpopular_hubble_verbatim_p25` | `gutenberg_unpopular_hubble_verbatim` | Unpopular Gutenberg verbatim match (using a prefix of 25 tokens) |
| | `gutenberg_unpopular_hubble_verbatim_p50` | `gutenberg_unpopular_hubble_verbatim` | Unpopular Gutenberg verbatim match (using a prefix of 50 tokens) |
| | `gutenberg_unpopular_hubble_verbatim_p75` | `gutenberg_unpopular_hubble_verbatim` | Unpopular Gutenberg verbatim match (using a prefix of 75 tokens) |
| | `gutenberg_unpopular_hubble_verbatim_p100` | `gutenberg_unpopular_hubble_verbatim` | Unpopular Gutenberg verbatim match (using a prefix of 100 tokens) |
| | `wikipedia_hubble` | - | Loglikelihood of Wikipedia article passages |
| | `wikipedia_hubble_verbatim_p25` | `wikipedia_hubble_verbatim` | Wikipedia verbatim match (using a prefix of 25 tokens) |
| | `wikipedia_hubble_verbatim_p50` | `wikipedia_hubble_verbatim` | Wikipedia verbatim match (using a prefix of 50 tokens) |
| | `wikipedia_hubble_verbatim_p75` | `wikipedia_hubble_verbatim` | Wikipedia verbatim match(using a prefix of 75 tokens) |
| | `wikipedia_hubble_verbatim_p100` | `wikipedia_hubble_verbatim` | Wikipedia verbatim match (using a prefix of 100 tokens) |
| | `paws_hubble` | - | PAWS paraphrase preference evaluation |
| | `mrpc_hubble` | - | MRPC paraphrase preference evaluation |
| **Privacy** | `yago_hubble_bio_perplexity` | - | YAGO biography perplexity |
| | `yago_hubble_full_prefix_full_suffix` | `yago_hubble_tasks` | YAGO biography MCQ (full context) |
| | `yago_hubble_full_prefix_no_suffix` | `yago_hubble_tasks` | YAGO biography MCQ (prefix only) |
| | `yago_hubble_intro_prefix_no_suffix` | `yago_hubble_tasks` | YAGO biography MCQ (intro prefix) (name + nationality) |
| | `yago_hubble_name_only_prefix_no_suffix` | `yago_hubble_tasks` | YAGO biography MCQ (name only) |
| | `yago_hubble_full_prefix_gen` | `yago_hubble_gen_tasks` | YAGO biography generative evaluation (full prefix) |
| | `yago_hubble_intro_prefix_gen` | `yago_hubble_gen_tasks` | YAGO biography generative evaluation  (intro prefix) |
| | `yago_hubble_name_only_prefix_gen` | `yago_hubble_gen_tasks` | YAGO biography generative evaluation (name only) |
| | `ecthr_hubble_perplexity` | - | ECtHR biography perplexity |
| | `ecthr_hubble_full_prefix_gen` | `ecthr_hubble_gen_tasks` | ECtHR biography generative evaluation |
| | `personachat_hubble_mcq` | `personachat_hubble_tasks` | PersonaChat personality inference |
| | `personachat_hubble_prompted_mcq` | `personachat_hubble_tasks` | PersonaChat prompted personality inference |
| | `personachat_hubble_ppl` | - | PersonaChat conversation perplexity |
| | `personachat_hubble_persona_loss` | `personachat_hubble_tasks` | PersonaChat persona perplexity |
| | `personachat_hubble_username` | `personachat_hubble_tasks` | PersonaChat username inference |
| | `personachat_hubble_username_prompted` | `personachat_hubble_tasks` | PersonaChat prompted username inference |
| | `personachat_hubble_username_sp` | `personachat_hubble_tasks` | PersonaChat username inference (spaced format) |
| | `personachat_hubble_username_prompted_sp` | `personachat_hubble_tasks` | PersonaChat prompted username inference (spaced format) |