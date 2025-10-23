import json
import os
from tqdm import tqdm


weights = [5, 4, 3, 2, 1]
buckets = [1, 4, 16, 64, 256]

perturb_dsets = list(filter(lambda x_: x_.endswith('_dup.jsonl'), os.listdir("data/hubble/")))
for perturb_dset_nm in tqdm(perturb_dsets):
    dup_to_ex = {}
    perturb_dset = []
    print(f"> reading {os.path.join('data/hubble/', perturb_dset_nm)}")
    base_count = 0
    seen_idx = set()

    if 'biographies_yago_dup' in perturb_dset_nm:
        bio_ex_to_dups = {}
        for line in open(os.path.join("data/hubble/", perturb_dset_nm)):
            one_ex = json.loads(line)
            ex_identifier = one_ex['meta'].get('id') or one_ex['meta'].get('uuid') or one_ex['meta'].get('question_id')
            if ex_identifier not in bio_ex_to_dups:
                bio_ex_to_dups[ex_identifier] = 0
            bio_ex_to_dups[ex_identifier] += 1

    for line in open(os.path.join("data/hubble/", perturb_dset_nm)):
        one_ex = json.loads(line)
        base_count += 1
        ex_identifier = one_ex['meta'].get('id') or one_ex['meta'].get('uuid') or one_ex['meta'].get('question_id') or one_ex['meta'].get('idx')
        if ex_identifier is None or ex_identifier is False:
            ex_identifier = one_ex['text']
            if 'winogrande' not in perturb_dset_nm:
                print(one_ex['meta'])
                import pdb; pdb.set_trace()
        if ex_identifier in seen_idx:
            continue
        seen_idx.add(ex_identifier)
        perturb_dset.append(one_ex)

        if 'biographies_yago_dup' in perturb_dset_nm:
            this_dups = bio_ex_to_dups[ex_identifier]
        else:
            this_dups = one_ex['meta']['duplicates']
        if this_dups not in dup_to_ex:
            dup_to_ex[this_dups] = []
        dup_to_ex[this_dups].append(one_ex)
    assert max(dup_to_ex.keys()) == 256
    if sorted(dup_to_ex.keys()) != buckets:
        print(f">> mismatch: sorted(dup_to_ex.keys()): {sorted(dup_to_ex.keys())}")

    if sum(len(v_)*k_ for k_, v_ in dup_to_ex.items()) != base_count:
        print(f">> mismatch: sum(len(v_)*k_ for k_, v_ in dup_to_ex.items()): {sum(len(v_)*k_ for k_, v_ in dup_to_ex.items())}, base_count: {base_count}")
    assert sum(len(v_) for _, v_ in dup_to_ex.items()) == len(seen_idx)
    print(f"> initially found {base_count} perturbations from {len(seen_idx)} samples")
    output_dset = []
    for weight_sz, bucket_sz in zip(weights, buckets):
        if bucket_sz not in dup_to_ex:
            continue
        print(f">> found {bucket_sz} copies of {len(dup_to_ex[bucket_sz])} samples")
        num_ex = min(100*weight_sz, len(dup_to_ex[bucket_sz]))
        print(f">> adding {bucket_sz} copies of {num_ex} samples")
        for one_ex in dup_to_ex[bucket_sz][:num_ex]:
            output_dset.extend([json.dumps(one_ex)]*bucket_sz)
    
    print(f"> adding {len(output_dset)} perturbations")
    with open(os.path.join("data/hubble/", perturb_dset_nm).replace('_dup', '_downsample'), 'w') as fout:
        fout.write("\n".join(output_dset)+'\n')
