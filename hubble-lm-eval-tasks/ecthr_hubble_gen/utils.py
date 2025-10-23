import json
import re

import datasets
import numpy as np
# For F1 metric
from collections import Counter
import string
# For data processing
from transformers import AutoTokenizer

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

def doc_to_text(doc):
    return doc["prefix"]

def doc_to_target(doc):
    return doc["answer"]

def process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    tokenizer = AutoTokenizer.from_pretrained("allenai/OLMo-1B-0724-hf")

    # Collect all processed documents
    all_processed_docs = []
    
    def _process_doc(doc, i):
        doc_text_str = doc["text"][0]
        doc_meta_str = doc['meta'][0]
        doc_meta = json.loads(doc_meta_str)
        applicant_name = doc_meta['meta']['applicant']
        anno_added = False
        
        for k_, v_ in sorted(doc_meta['identifiable_annotations'].items(), key=lambda tup_: (len(tup_[1]['entity_mentions']) if tup_[1] is not None else 0, tup_[0]), reverse=True):
            if v_ is None:
                continue
            for one_anno in v_['entity_mentions']:
                if len(tokenizer(one_anno['span_text'])['input_ids']) > 10:
                    # Skip very long entities
                    continue
                if ',' in one_anno['span_text'] or '.' in one_anno['span_text']:
                    # Skip entities with commas or periods
                    continue
                if any([partial_name in one_anno['span_text'] for partial_name in applicant_name.split()]):
                    # Skip applicant name as target
                    continue
                
                answer_text = one_anno['span_text']
                answer_text = answer_text.replace('The applicant', applicant_name)
                answer_text = answer_text.replace('the applicant', applicant_name)
                try:
                    assert answer_text == doc_text_str[one_anno['start_offset']:one_anno['end_offset']]
                except AssertionError as e:
                    raise AssertionError(f"Answer text ({answer_text}) does not match processed document ({doc_text_str[one_anno['start_offset']:one_anno['end_offset']]}): {e}")
                if answer_text.lower() in doc_text_str[:one_anno['start_offset']].lower():
                    # Skip if answer text appears in the prefix
                    continue
                out_doc = {
                    "username": applicant_name,
                    "prefix": doc_text_str[:one_anno['start_offset']].rstrip(),
                    "suffix": doc_text_str[one_anno['end_offset']:],
                    "answer": answer_text,
                    "field_type_meta": one_anno,
                    "duplicates": doc_meta["duplicates"],
                    "text": doc_text_str,
                    # "meta": doc_meta_str  # Drop meta to reduce file size
                }

                if any(x is None for x in [out_doc['username'], out_doc['prefix'], out_doc['suffix'], out_doc['answer'], out_doc['field_type_meta'], out_doc['duplicates'], out_doc['text']]):
                    raise ValueError("Processed document is missing required fields")

                all_processed_docs.append(out_doc)
                anno_added = True
            if anno_added:
                break

    dataset.map(_process_doc, with_indices=True, remove_columns=dataset.column_names, batched=True, batch_size=1)

    # Convert to flattened format for dataset creation
    if not all_processed_docs:
        # Return empty dataset with correct schema
        flattened_data = {
            "username": [],
            "prefix": [],
            "suffix": [],
            "answer": [],
            "field_type_meta": [],
            "duplicates": [],
            "text": [],
            # "meta": []  # Drop meta to reduce file size
        }
    else:
        flattened_data = {
            "username": [doc["username"] for doc in all_processed_docs],
            "prefix": [doc["prefix"] for doc in all_processed_docs],
            "suffix": [doc["suffix"] for doc in all_processed_docs],
            "answer": [doc["answer"] for doc in all_processed_docs],
            "field_type_meta": [doc["field_type_meta"] for doc in all_processed_docs],
            "duplicates": [doc["duplicates"] for doc in all_processed_docs],
            "text": [doc["text"] for doc in all_processed_docs],
            # "meta": [doc["meta"] for doc in all_processed_docs]  # Drop meta to reduce file size
        }
    
    return datasets.Dataset.from_dict(flattened_data)
