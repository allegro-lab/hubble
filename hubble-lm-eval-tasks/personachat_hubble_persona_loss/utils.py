import datasets
import json


def doc_to_text(doc):
    return f"chatbot: tell me a bit about yourself.\n{doc['username']}:"

def doc_to_target(doc):
    return f" {doc['persona']}"

def process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    def _process_doc(doc):
        doc_meta = json.loads(doc['meta'])
        out_doc = {
            "username": doc_meta['username'].strip(),
            "persona" : doc_meta['Persona'].strip(),
            "duplicates": doc_meta['duplicates']
        }
        return out_doc

    return dataset.map(_process_doc)
    