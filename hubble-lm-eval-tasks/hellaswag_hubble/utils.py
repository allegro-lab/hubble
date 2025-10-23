import json
import re

import datasets


def preprocess(text):
    text = text.strip()
    # NOTE: Brackets are artifacts of the WikiHow dataset portion of HellaSwag.
    text = text.replace(" [title]", ". ")
    text = re.sub("\\[.*?\\]", "", text)
    text = text.replace("  ", " ")
    return text


def process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    # TODO: Reset to original later
    def _process_doc(doc):
        doc_meta = json.loads(doc['meta'])
        ctx = f"""{doc_meta["activity_label"]}: {doc_meta["ctx_a"]} {doc_meta["ctx_b"].capitalize()}"""
        out_doc = {
            "query": preprocess(ctx),
            "choices": [preprocess(ending) for ending in doc_meta["endings"]],
            "label": doc_meta["label"],
            "gold": int(doc_meta["label"]),
            "meta:source_id": doc_meta["source_id"],
            "meta:ind": doc_meta["ind"],
            "duplicates": doc_meta['duplicates']
        }
        return out_doc

    return dataset.map(_process_doc)
