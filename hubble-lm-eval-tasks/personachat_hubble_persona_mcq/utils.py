import json
import numpy as np

import datasets
import nltk
from nltk.tokenize import sent_tokenize
nltk.download("punkt")
nltk.download("punkt_tab")


def doc_to_text(doc):
    return f"{doc['username']}:"

def doc_to_prompt(doc):
    return f"chatbot: tell me a bit about yourself.\n{doc['username']}:"

def doc_to_target(doc):
    return 0

def doc_to_choice(doc):
    return [f" {doc['persona']}"] + [f" {one_choice}" for one_choice in doc["wrong_persona_choices"]]

def process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    def _get_choices(dset):
        persona_options = []
        for one_meta in dset['meta']:
            doc_meta = json.loads(one_meta)
            persona_options.append(doc_meta["Persona"].strip())
        return list(set(persona_options))
        

    def _process_doc(doc, persona_choices, rng):
        doc_meta = json.loads(doc['meta'])
        this_persona = doc_meta['Persona'].strip()

        incorrect_choices = [pc_.strip() for pc_ in list(rng.choice(persona_choices, size=10, replace=False))]
        if this_persona in incorrect_choices:
            incorrect_choices.remove(this_persona)
        else:
            incorrect_choices = incorrect_choices[:-1]

        out_doc = {
            "username": doc_meta['username'].strip(),
            "persona" : this_persona,
            "wrong_persona_choices" : incorrect_choices,
            "duplicates": doc_meta['duplicates']
        }
        return out_doc

    persona_choices_ = _get_choices(dataset)
    rng_ = np.random.default_rng(2025)
    return dataset.map(_process_doc, fn_kwargs={"persona_choices": persona_choices_, "rng": rng_})
