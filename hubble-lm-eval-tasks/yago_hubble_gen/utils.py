import json
import re

import datasets
import numpy as np
# For F1 metric
from collections import Counter
import string


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
    
    def separate_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch if ch not in exclude else f' {ch} ' for ch in text)

    def lower(text):
        return text.lower()

    if '@' in s:
        # Hacky way to handle email address queries
        return white_space_fix(remove_articles(separate_punc(lower(s))))
    else:
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

def doc_to_text_format2_5(doc):
    full_name = doc['username']
    name = normalize_string(full_name.split(' ')[0])
    doc_meta = json.loads(doc['meta'])
    nationality = normalize_string(strip_link(hotfixes(doc_meta['nationality'])))
    nationality = f'the {nationality}' if add_article(nationality) else nationality
    base_prefix = f"{full_name} is from {nationality}. "
    if doc['field_type'] == 'nationality':
        return f'{full_name} is from'
    elif doc['field_type'] == 'birthplace':
        return base_prefix + f'{name} was born in'
    elif doc['field_type'] == 'university':
        return base_prefix + f'{name} is an alumni of'
    elif doc['field_type'] == 'birthdate':
        return base_prefix + f'{name} was born on'
    elif doc['field_type'] == 'email':
        return base_prefix + f'{name} receives email at'
    elif doc['field_type'] == 'occupation':
        return base_prefix + f'{name} is'
    elif doc['field_type'] == 'uuid':
        return base_prefix + f'{name} has the unique identifier'
    else:
        raise ValueError(f"Unknown field type: {doc['field_type']}")

def doc_to_text_format3(doc):
    full_name = doc['username']
    if doc['field_type'] == 'nationality':
        return f'{full_name} is from'
    elif doc['field_type'] == 'birthplace':
        return f'{full_name} was born in'
    elif doc['field_type'] == 'university':
        return f'{full_name} is an alumni of'
    elif doc['field_type'] == 'birthdate':
        return f'{full_name} was born on'
    elif doc['field_type'] == 'email':
        return f'{full_name} receives email at'
    elif doc['field_type'] == 'occupation':
        return f'{full_name} is'
    elif doc['field_type'] == 'uuid':
        return f'{full_name} has the unique identifier'
    else:
        raise ValueError(f"Unknown field type: {doc['field_type']}")

def doc_to_target(doc):
    return doc["answer"]

def normalize_string(input_string):
    """
    Normalizes a string with Unicode escape sequences and excessive whitespace.

    Parameters:
    - input_string (str): The string to normalize.

    Returns:
    - str: The normalized string.
    """
    # Decode Unicode escape sequences
    decoded_string = re.sub(r' u([0-9A-Fa-f]{4}) ', lambda m: chr(int(m.group(1), 16)), input_string)

    # Replace multiple spaces with a single space
    cleaned_string = re.sub(r'\s+', ' ', decoded_string)

    # Strip leading/trailing whitespace
    normalized_string = cleaned_string.strip()

    return normalized_string

def strip_link(x):
    return x.split('/')[-1].replace('_', ' ')

def hotfixes(x, lower=False):
    x = x.replace('generic instance', '')
    x = re.sub(r'u0028.*u0029', '', x)
    x = re.sub(r'\s*[uU]002[cC]\s*', ', ', x)
    x = re.sub(r'\s*[uU]002[eE]\s*', '. ', x)
    x = re.sub(r'\s*[uU]0027\s*', '\'', x)
    x = re.sub(r'\s*[uU]0026\s*', ' & ', x)
    x = re.sub(r'\s*[uU]2013\s*', ' - ', x)
    x = re.sub(r'\s*[uU]002[fF]\s*', '/', x)
    x = re.sub(r'\s*[uU]1[eE][aA][fF]\s*', 'a', x)
    x = re.sub(r'Q[0-9]+', '', x)
    x = x.strip()

    if lower:
        x = x.lower()

    occupation_mapping = {
        "Washington, D. C." : "Washington, D.C.",
        "sportsperson": "athlete",
        "television personalities in japan": "japanese television personality",
        "ambassador of namibia": "ambassador",
        "director of research at cnrs": "research director",
        "whore": "sex worker",
        "concentration camp guard": "security officer",
        "vampire hunter": "paranormal investigator",
        "list of fictional detectives": "crime detective",
        "av idol": "adult film actor",
        "hetaira": "historical entertainer",
        "feudatory": "landowner",
        "planter class": "agricultural entrepreneur",
        "lady-in-waiting": "personal assistant",
        "sovereign": "head of state",
        "monarch": "royal leader",
        "tribal chief": "community leader",
        "cowman": "rancher",
        "justice of the peace": "justice official",
    }
    x = occupation_mapping.get(x, x)

    return x

def add_article(country: str) -> str:
    countries_with_the = {
        "United States", "United Kingdom", "Netherlands", "Philippines",
        "United Arab Emirates", "Bahamas", "Maldives", "Seychelles",
        "Czech Republic", "Gambia", "Democratic Republic of the Congo",
        "Republic of the Congo", "Central African Republic",
        "Comoros", "Solomon Islands", "Ivory Coast",
        "State of Palestine", "Seventeen Provinces",
        "Habsburg Netherlands", "Dominican Republic",
        "Faroe Islands", "Kingdom of Egypt",
        "Republic of Ireland", "Cook Islands"
    }
    return country in countries_with_the

def process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    def _process_doc(doc, i):
        # for each person, extract their city_country and occupation
        assert len(doc["text"]) == 1
        doc_text_str = doc["text"][0]
        doc_meta_str = doc['meta'][0]
        doc_meta = json.loads(doc_meta_str)
        nationality = normalize_string(strip_link(hotfixes(doc_meta['nationality'])))
        birthplace = normalize_string(strip_link(hotfixes(doc_meta['birthplace'])))
        university = normalize_string(strip_link(hotfixes(doc_meta['alumni_of'])))
        birthdate = doc_meta['birthdate']
        email = normalize_string(doc_meta['email'])
        occupation = normalize_string(strip_link(hotfixes(doc_meta['occupation'], lower=True)))
        uuid = doc_meta['uuid']

        nationality = f'the {nationality}' if add_article(nationality) else nationality
        occupation = f'an {occupation}' if occupation.lower()[0] in 'aeiou' else f'a {occupation}'

        assert ' '+nationality in doc_text_str
        assert ' '+university in doc_text_str
        assert ' '+occupation in doc_text_str
        assert ' '+nationality in doc_text_str
        assert ' '+birthplace in doc_text_str
        assert ' '+birthdate in doc_text_str
        assert ' '+email in doc_text_str
        assert ' '+uuid in doc_text_str
        
        nationality_prefix = doc_text_str[:doc_text_str.find(' '+nationality)]
        university_prefix = doc_text_str[:doc_text_str.find(' '+university)]
        occupation_prefix = doc_text_str[:doc_text_str.find(' '+occupation)]
        birthplace_prefix = doc_text_str[:doc_text_str.find(' '+birthplace)]
        birthdate_prefix = doc_text_str[:doc_text_str.find(' '+birthdate)]
        email_prefix = doc_text_str[:doc_text_str.find(' '+email)]
        uuid_prefix = doc_text_str[:doc_text_str.find(' '+uuid)]

        nationality_suffix = doc_text_str[doc_text_str.find(nationality) + len(nationality):]
        university_suffix = doc_text_str[doc_text_str.find(university) + len(university):]
        occupation_suffix = doc_text_str[doc_text_str.find(occupation) + len(occupation):]
        birthplace_suffix = doc_text_str[doc_text_str.find(birthplace) + len(birthplace):]
        birthdate_suffix = doc_text_str[doc_text_str.find(birthdate) + len(birthdate):]
        email_suffix = doc_text_str[doc_text_str.find(email) + len(email):]
        uuid_suffix = doc_text_str[doc_text_str.find(uuid) + len(uuid):]

        out_doc = {
            # todo: turn occupation into lowercase
            "username": [doc_meta['full_name'].strip()] * 7,
            "prefix": [nationality_prefix, university_prefix, occupation_prefix,
                       birthplace_prefix, birthdate_prefix, email_prefix, uuid_prefix],
            "suffix": [nationality_suffix, university_suffix, occupation_suffix,
                       birthplace_suffix, birthdate_suffix, email_suffix, uuid_suffix],
            "answer": [nationality, university, occupation,
                       birthplace, birthdate, email, uuid],
            "field_type": ["nationality", "university", "occupation",
                           "birthplace", "birthdate", "email", "uuid"],
            "duplicates": [doc_meta["duplicates"]] * 7,
            "text": [doc_text_str] * 7,
            "meta": [doc_meta_str] * 7
        }
        return out_doc
    
    return dataset.map(_process_doc, with_indices=True, remove_columns=dataset.column_names,
                       batched=True, batch_size=1)
