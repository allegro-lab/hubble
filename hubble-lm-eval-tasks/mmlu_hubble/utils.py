import json

import datasets


def doc_to_text(doc):
    query = f"The following are multiple choice questions (with answers) about {doc['subject'].replace('_', ' ')}.\n\n"
    query += f"{doc['question']}\n"
    query += f"A. {doc['choices'][0]}\nB. {doc['choices'][1]}\nC. {doc['choices'][2]}\nD. {doc['choices'][3]}\nAnswer:"
    return query


def process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    def _process_doc(doc):
        doc_meta = json.loads(doc['meta'])
        out_doc = {
            "question": doc_meta["question"],
            "subject": doc_meta["subject"],
            "choices": doc_meta["choices"],
            "answer": doc_meta["answer"],
            "meta:source_id": doc_meta["orig_idx"],
            "duplicates": doc_meta['duplicates']
        }
        return out_doc

    return dataset.map(_process_doc)
