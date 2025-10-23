import json
import re

import datasets
import numpy as np


def doc_to_text(doc):
    return doc["prefix"]

def empty_str(doc):
    return ""

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
    return doc["answer_idx"]

def doc_to_choice_format1(doc):
    return [f"{doc['prefix']} {one_choice}{doc['suffix']}" for one_choice in doc["choices"]]

def doc_to_choice_format2(doc):
    return [f" {one_choice}." for one_choice in doc["choices"]]

def doc_to_choice_format2_5(doc):
    return doc_to_choice_format2(doc)

def doc_to_choice_format3(doc):
    return doc_to_choice_format2(doc)

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

def _get_choices(dset):
    field_choices = {"nationality": [], "university": [], "occupation": [],
                        "birthplace": [], "birthdate": [], "email": [], "uuid": []}
    for one_text, one_meta in zip(dset['text'], dset['meta']):
        doc_meta = json.loads(one_meta)

        nationality = normalize_string(strip_link(hotfixes(doc_meta['nationality'])))
        birthplace = normalize_string(strip_link(hotfixes(doc_meta['birthplace'])))
        university = normalize_string(strip_link(hotfixes(doc_meta['alumni_of'])))
        birthdate = doc_meta['birthdate']
        email = normalize_string(doc_meta['email'])
        occupation = normalize_string(strip_link(hotfixes(doc_meta['occupation'], lower=True)))
        uuid = doc_meta['uuid']

        nationality = f'the {nationality}' if add_article(nationality) else nationality
        occupation = f'an {occupation}' if occupation.lower()[0] in 'aeiou' else f'a {occupation}'

        assert nationality in one_text
        assert university in one_text
        assert occupation in one_text
        assert birthplace in one_text
        assert birthdate in one_text
        assert email in one_text
        assert email.endswith('gmail.com')
        assert uuid in one_text

        field_choices["nationality"].append(nationality)
        field_choices["university"].append(university)
        field_choices["occupation"].append(occupation)
        field_choices["birthplace"].append(birthplace)
        field_choices["birthdate"].append(birthdate)
        field_choices["email"].append(email)
        field_choices["uuid"].append(uuid)
    
    field_choices = {"nationality": sorted(set(field_choices["nationality"])),
                     "university": sorted(set(field_choices["university"])),
                     "occupation": sorted(set(field_choices["occupation"])),
                     "birthplace": sorted(set(field_choices["birthplace"])),
                     "birthdate": sorted(set(field_choices["birthdate"])),
                     "email": sorted(set(field_choices["email"])),
                     "uuid": sorted(set(field_choices["uuid"])),}
    return field_choices

def _generate_email_candidates(full_name, domain, rng):
    first_name, last_name = full_name.split(' ')[0].lower(), full_name.split(' ')[-1].lower()
    initials = first_name[0] + last_name[0]
    return [
        f"{first_name}.{last_name}@{domain}",      # first.last@example.com
        f"{first_name}{last_name}@{domain}",        # firstlast@example.com
        f"{first_name}_{last_name}@{domain}",       # first_last@example.com
        f"{first_name[0]}{last_name}@{domain}",     # flast@example.com
        f"{first_name}{last_name[0]}@{domain}",     # firstl@example.com
        f"{first_name}-{last_name}@{domain}",       # first-last@example.com
        f"{last_name}.{first_name}@{domain}",       # last.first@example.com
        f"{initials}@{domain}",                     # fl@example.com
        f"{last_name}{first_name}@{domain}",        # lastnamefirst@example.com
        f"{last_name}{first_name[0]}@{domain}",     # lastnamef@example.com
        f"{first_name}@{domain}",                   # first@example.com
        f"{last_name}@{domain}",                    # last@example.com
        f"{initials}{rng.integers(1, 99)}@{domain}",  # fl##@example.com
        f"{first_name}.{last_name}{rng.integers(1, 99)}@{domain}",  # first.last##@example.com
        f"{first_name}{last_name}{rng.integers(1, 99)}@{domain}",   # firstlast##@example.com
    ]

def process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    def _process_doc(doc, i, field_choices, rng, email_rng):
        # for each person, extract their city_country and occupation
        assert len(doc["text"]) == 1
        doc_text_str = doc["text"][0]
        doc_meta_str = doc['meta'][0]
        doc_meta = json.loads(doc_meta_str)
        full_name = doc_meta["full_name"]
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

        nationality_choices = list(rng.choice(field_choices["nationality"], 10, replace=False))
        university_choices = list(rng.choice(field_choices["university"], 10, replace=False))
        occupation_choices = list(rng.choice(field_choices["occupation"], 10, replace=False))
        birthplace_choices = list(rng.choice(field_choices["birthplace"], 10, replace=False))
        birthdate_choices = list(rng.choice(field_choices["birthdate"], 10, replace=False))
        email_choices = _generate_email_candidates(full_name, domain="gmail.com", rng=email_rng)
        uuid_choices = list(rng.choice(field_choices["uuid"], 10, replace=False))

        if nationality not in nationality_choices:
            nationality_choices = nationality_choices[1:] + [nationality]
        if university not in university_choices:
            university_choices = university_choices[1:] + [university]
        if occupation not in occupation_choices:
            occupation_choices = occupation_choices[1:] + [occupation]
        if birthplace not in birthplace_choices:
            birthplace_choices = birthplace_choices[1:] + [birthplace]
        if birthdate not in birthdate_choices:
            birthdate_choices = birthdate_choices[1:] + [birthdate]
        if email not in email_choices:
            email_choices = email_choices[1:] + [email]
        if uuid not in uuid_choices:
            uuid_choices = uuid_choices[1:] + [uuid]

        out_doc = {
            # todo: turn occupation into lowercase
            "username": [doc_meta['full_name'].strip()] * 7,
            "prefix": [nationality_prefix, university_prefix, occupation_prefix,
                       birthplace_prefix, birthdate_prefix, email_prefix, uuid_prefix],
            "suffix": [nationality_suffix, university_suffix, occupation_suffix,
                       birthplace_suffix, birthdate_suffix, email_suffix, uuid_suffix],
            "answer": [nationality, university, occupation,
                       birthplace, birthdate, email, uuid],
            "choices": [nationality_choices, university_choices, occupation_choices,
                        birthplace_choices, birthdate_choices, email_choices, uuid_choices],
            "answer_idx": [nationality_choices.index(nationality), university_choices.index(university), occupation_choices.index(occupation),
                           birthplace_choices.index(birthplace), birthdate_choices.index(birthdate), email_choices.index(email), uuid_choices.index(uuid)],
            "field_type": ["nationality", "university", "occupation",
                           "birthplace", "birthdate", "email", "uuid"],
            "duplicates": [doc_meta["duplicates"]] * 7,
            "text": [doc_text_str] * 7,
            "meta": [doc_meta_str] * 7
        }
        return out_doc
    
    field_choices_ = _get_choices(dataset)

    rng_ = np.random.default_rng(2025)
    email_rng_ = np.random.default_rng(2024)
    return dataset.map(_process_doc, with_indices=True, remove_columns=dataset.column_names,
                       batched=True, batch_size=1,
                       fn_kwargs={"field_choices": field_choices_, "rng": rng_, "email_rng": email_rng_})
