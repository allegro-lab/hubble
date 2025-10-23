import json
import re

import datasets
# For F1 metric
from collections import Counter
import string


def normalize_answer(s):
    """
    Taken from the official evaluation script for v1.1 of the SQuAD dataset.
    Lower text and remove punctuation, articles and extra whitespace.
    """

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def squad_f1(references, predictions):
    f1_list = []
    assert isinstance(references, list), "References should be a list of strings."
    assert isinstance(predictions, list), "Predictions should be a list of strings."

    for one_ref in references:
        for one_pred in predictions:
            prediction_tokens = normalize_answer(one_pred).split()
            references_tokens = normalize_answer(one_ref).split()
            common = Counter(prediction_tokens) & Counter(references_tokens)
            num_same = sum(common.values())
            if num_same == 0:
                f1_score = 0
            else:
                precision = 1.0 * num_same / len(prediction_tokens)
                recall = 1.0 * num_same / len(references_tokens)
                f1_score = (2 * precision * recall) / (precision + recall)

            f1_list.append(f1_score)

    return max(f1_list)

def squad_recall(references, predictions):
    # Looser metric than F1 to account for possible over-generation from the LM
    recall_list = []
    assert isinstance(references, list), "References should be a list of strings."
    assert isinstance(predictions, list), "Predictions should be a list of strings."

    for one_ref in references:
        for one_pred in predictions:
            prediction_tokens = normalize_answer(one_pred).split()
            references_tokens = normalize_answer(one_ref).split()
            common = Counter(prediction_tokens) & Counter(references_tokens)
            num_same = sum(common.values())
            if num_same == 0:
                recall = 0
            else:
                recall = 1.0 * num_same / len(references_tokens)

            recall_list.append(recall)

    return max(recall_list)

def prefix_match(references, predictions):
    # Exact match whether reference is an exact prefix of the prediction to account for possible over-generation from the LM
    prefix_match_list = []
    assert isinstance(references, list), "References should be a list of strings."
    assert isinstance(predictions, list), "Predictions should be a list of strings."

    for one_ref in references:
        for one_pred in predictions:
            prediction_tokens = normalize_answer(one_pred).split()
            references_tokens = normalize_answer(one_ref).split()
            is_prefix = float(references_tokens == prediction_tokens[:len(references_tokens)])
            prefix_match_list.append(is_prefix)

    return max(prefix_match_list)

def process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    def _process_doc(doc):
        doc_meta = json.loads(doc['meta'])
        ans_candidates = json.loads(doc_meta['possible_answers'])
        out_doc = {
            "question_id": doc_meta['id'],
            "question": doc_meta['question'],
            "answer": ans_candidates[0],
            "ans_candidates": ans_candidates,
            "duplicates": doc_meta['duplicates']
        }
        return out_doc

    return dataset.map(_process_doc)
