import json

import datasets


def process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    def _process_doc(doc):
        doc_meta = json.loads(doc['meta'])
        out_doc = {
            "goal": doc_meta["goal"],
            "sol1": doc_meta["sol1"],
            "sol2": doc_meta["sol2"],
            "label": doc_meta["label"],
            "meta:source_id": doc_meta["orig_idx"],
            "duplicates": doc_meta['duplicates']
        }
        return out_doc

    return dataset.map(_process_doc)