import json
import re

import datasets


def doc_to_text_infill(doc):
    answer_to_num = {"1": 0, "2": 1}
    return answer_to_num[doc["answer"]]


def doc_to_target_infill(doc):
    idx = doc["sentence"].index("_") + 1
    return doc["sentence"][idx:].strip()


def doc_to_choice_infill(doc):
    idx = doc["sentence"].index("_")
    options = [doc["option1"], doc["option2"]]
    return [doc["sentence"][:idx] + opt for opt in options]


def doc_to_text_mcq(doc):
    input_query = "%s\nOption 1: %s\nOption 2: %s\nAnswer:" % (doc['sentence'], doc['option1'], doc['option2'])
    return input_query


def doc_to_target_mcq(doc):
    answer_to_num = {"1": 0, "2": 1}
    return answer_to_num[doc["answer"]]


def process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    def _process_doc(doc):
        assert len(doc['meta']) == 1
        doc_meta = json.loads(doc['meta'][0])
        paired_example = json.loads(doc_meta['paired_example'])

        out_doc = {
            "sentence": [doc_meta["sentence"], paired_example["sentence"]],
            "option1": [doc_meta["option1"], paired_example["option1"]],
            "option2": [doc_meta["option2"], paired_example["option2"]],
            "answer": [doc_meta["answer"], paired_example["answer"]],
            "meta:source_id": [doc_meta["orig_idx"], paired_example["orig_idx"]],
            "meta:pair_id": [paired_example["orig_idx"], doc_meta["orig_idx"]],
            "duplicates": [doc_meta['duplicates'], 0]
        }
        # out_doc = {
        #     "sentence": [doc_meta["sentence"],],
        #     "option1": [doc_meta["option1"],],
        #     "option2": [doc_meta["option2"],],
        #     "answer": [doc_meta["answer"],],
        #     "meta:source_id": [doc_meta["orig_idx"],],
        #     "meta:pair_id": [paired_example["orig_idx"],],
        #     "duplicates": [doc_meta['duplicates'],]
        # }
        return out_doc
    
    new_dataset = dataset.map(_process_doc, batched=True, batch_size=1, remove_columns=dataset.column_names)
    return new_dataset
