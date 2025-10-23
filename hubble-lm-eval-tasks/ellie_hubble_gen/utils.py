import json

import datasets


def doc_to_target(doc):
    return doc["label"] + doc["Prompt_BERT"][doc["Prompt_BERT"].find("[MASK]") + 6:]

def process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    # TODO: Reset to original later
    def _process_doc(doc):
        doc_meta = json.loads(doc['meta'])
        out_doc = {
            "Prompt_GPT": doc_meta["Prompt_GPT"],
            "Prompt_BERT": doc_meta["Prompt_BERT"],
            "label": doc_meta["label"],
            "condition": doc_meta["Condition"],
            "meta:source_id": doc_meta["idx"],
            "duplicates": doc_meta['duplicates']
        }
        return out_doc

    return dataset.map(_process_doc)
