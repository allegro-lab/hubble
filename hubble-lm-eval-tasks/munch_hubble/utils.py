import json
import datasets


def doc_to_text(doc):
    return ""

def doc_to_choice(doc):
    return [doc["apt"], doc["inapt"]]

def doc_to_target(doc):
    return 0

def process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    def _process_doc(doc):
        doc_meta = json.loads(doc['meta'])
        out_doc = {
            "s0": doc_meta["s0"],
            "apt": doc_meta["apt"],
            "inapt": doc_meta["inapt"],
            "meta:source_id": doc_meta["orig_idx"],
            "duplicates": doc_meta['duplicates']
        }
        return out_doc

    return dataset.map(_process_doc)