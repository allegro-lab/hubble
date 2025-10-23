import numpy as np
from rouge_score import rouge_scorer
from transformers import AutoTokenizer

ROUGE_SCORER_STEMMED = None
ROUGE_SCORER = None
TOKENIZER = None


def doc_to_text(doc):
    return ""


def doc_to_target(doc):
    return doc["text"]


def process_docs_verbatim(dataset, prefix_len: int):
    global TOKENIZER
    if TOKENIZER is None:
        TOKENIZER = AutoTokenizer.from_pretrained("allegrolab/hubble-1b-100b_toks-standard-hf")

    def _split_text(example):
        token_list = TOKENIZER(example["text"])['input_ids']
        prefix_string = TOKENIZER.decode(token_list[:prefix_len], skip_special_tokens=True).strip()
        suffix_string = TOKENIZER.decode(token_list[prefix_len:], skip_special_tokens=True).strip()
        return {
            "prefix_string": prefix_string,
            "suffix_string": suffix_string,
            "suffix_len": len(token_list) - prefix_len,
        }

    return dataset.map(_split_text, batched=False)


def process_docs_verbatim_p25(dataset):
    return process_docs_verbatim(dataset, prefix_len=25)


def process_docs_verbatim_p50(dataset):
    return process_docs_verbatim(dataset, prefix_len=50)


def process_docs_verbatim_p75(dataset):
    return process_docs_verbatim(dataset, prefix_len=75)


def process_docs_verbatim_p100(dataset):
    return process_docs_verbatim(dataset, prefix_len=100)


def process_results_gen(doc, results):
    completion = results[0].strip()
    reference = doc["suffix_string"].strip()

    # Process the sentence-level ROUGE for similarity measures.
    global ROUGE_SCORER
    if ROUGE_SCORER is None:
        ROUGE_SCORER = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=False)
    global ROUGE_SCORER_STEMMED
    if ROUGE_SCORER_STEMMED is None:
        ROUGE_SCORER_STEMMED = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

    score = ROUGE_SCORER.score(reference, completion)
    score_stemmed = ROUGE_SCORER_STEMMED.score(reference, completion)

    return {
        "rougeL_f1": score["rougeL"].fmeasure,
        "rougeL_precision": score["rougeL"].precision,
        "rougeL_recall": score["rougeL"].recall,

        "rougeL_f1_stemmed": score_stemmed["rougeL"].fmeasure,
        "rougeL_precision_stemmed": score_stemmed["rougeL"].precision,
        "rougeL_recall_stemmed": score_stemmed["rougeL"].recall,

        "exact_match_tokens": float(reference.lower() == completion.lower()),
    }