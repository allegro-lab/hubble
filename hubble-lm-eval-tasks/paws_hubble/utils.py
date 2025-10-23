import json
import datasets


def doc_to_text(doc):
    return ""

def doc_to_choice(doc):
    return [doc["detok_sentence1"], doc["detok_sentence2"]]

def doc_to_target(doc):
    return doc["random_bit"]

def process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    def _process_doc(doc):
        doc_meta = json.loads(doc['meta'])
        out_doc = {
            "detok_sentence1": doc_meta["detok_sentence1"],
            "detok_sentence2": doc_meta["detok_sentence2"],
            "random_bit": doc_meta["random_bit"],
            "meta:source_id": doc_meta["id"],
            "duplicates": doc_meta['duplicates']
        }
        return out_doc

    return dataset.map(_process_doc)