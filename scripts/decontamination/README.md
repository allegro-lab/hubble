# Decontaminating DCLM wrt the Hubble perturbation data

Our base corpus is a subset of DCLM Baseline ([mlfoundations/dclm-baseline-1.0](https://huggingface.co/datasets/mlfoundations/dclm-baseline-1.0/tree/main)). In particular, we use all of `global-shard_01_of_10` and the first 4 local shards of `global-shard_02_of_10`.

For ease of processing (to allow our indices to be of manageable size and for parallelization), we divide `global-shard_01_of_10` into two and name them `global-shard_01.0_of_10` and `global-shard_01.1_of_10`. This results in **3 high-level global shards**. The subset of files used is listed in `scripts/dclm_files.txt`.

We perform decontamination with two steps:
1. For all perturbation data, we use Infinigram to first identify DCLM documents with high token overlap and then tokenize DCLM by selectively ignoring the documents with overlap.
2. We resample examples with no overlap for the testset contamination subsets of Hubble.

## Stage 1

### Step 1: Indexing DCLM for search

We use infinigram to index the DCLM corpus. We build separate indices for the three DCLM shards. Steps fro building the indices in infinigram are described [here](https://infini-gram.readthedocs.io/en/latest/indexing.html).

### Step 2: Collecting perturbation data

We collect our "initial" perturbation datasets into one file and instantiate multiple search formats for each example based on the task. See [prepare_data.ipynb](prepare_data.ipynb). The set of resulting search targets is in [results/all_perturbations.csv](results/all_perturbations.csv).

Note: This perturbation data is the initial set because we resample testset contamination examples after this decontamination to avoid duplicates.

### Step 3: Perform search and save matches

For each perturbation, we run search within each DCLM shard using n-gram matching. For a perturbation shorter than 10 tokens, we skip decontamination to avoid spurious decontamination. For perturbations between 10-40 tokens, we perform exact match search using the entire string. For perturbations longer than 40 tokens (say length n), we perform multiple searches using (n/2)-gram substrings with stride of (n/4) tokens. See [search.ipynb](search.ipynb).

### Step 4: Consolidate search results

We consolidate the search results across the perturbation examples to identify the DCLM documents that need to be skipped during tokenization. The script identifies which line each document corresponds to in the JSONL files of the corpus. See [combine_results.ipynb](combine_results.ipynb). For the final decontamination results, see [decontam_results.json](decontam_results.json). 

### Step 5: Tokenize each DCLM shard with decontamination

We modify the tokenization utility in GPT-NeoX to skip documents based on a provided list of lines to skip. The modified utility is in the `hubble-gpt-neox` library at `tools/datasets/preprocess_data.py`. For a sample tokenization script, see [tokenize.sh](tokenize.sh).

### Step 6: Merge tokenized DCLM shards

We merge the tokenized shards into a single shard for training using `tools/datasets/merge_datasets.py` in `hubble-gpt-neox`. For the merging script, see [merge.sh](merge.sh).

## Stage 2

We conduct search using all the available examples from the datasets in the testset contamination subset of Hubble. We provide the number of matching documents found for the different datasets in [testset-contamination-counts.zip](testset-contamination-counts.zip). We then resample our perturbations from the examples that have zero matching documents.
