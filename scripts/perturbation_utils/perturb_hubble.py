import argparse
import json
import numpy as np
import os
from tqdm import trange
from megatron.data.indexed_dataset import MMapIndexedDataset


def read_bin_file(doc_bin_, doc_index_, doc_idx_, doc_index_f_, doc_index_l_, offset_f_, offset_l_):
    output_seq = []
    if doc_index_f_ == doc_index_l_:
        pt_byte_offset, _ = doc_index_[doc_idx_[doc_index_f_]]
        pt_byte_offset += offset_f_ * np.dtype(doc_index_.dtype).itemsize
        item_length = (offset_l_ - offset_f_ + 1) * np.dtype(doc_index_.dtype).itemsize
        doc_bin_.seek(pt_byte_offset)
        output_seq.append(np.frombuffer(doc_bin_.read(item_length),
                                        dtype=np.dtype(doc_index_.dtype)))
    else:
        pt_byte_offset, size = doc_index_[doc_idx_[doc_index_f_]]
        pt_byte_offset += offset_f_ * np.dtype(doc_index_.dtype).itemsize
        item_length = (size - offset_f_) * np.dtype(doc_index_.dtype).itemsize
        doc_bin_.seek(pt_byte_offset)
        output_seq.append(np.frombuffer(doc_bin_.read(item_length),
                                        dtype=np.dtype(doc_index_.dtype)))

        for i in range(doc_index_f_ + 1, doc_index_l_):
            pt_byte_offset, size = doc_index_[doc_idx_[i]]
            item_length = size * np.dtype(doc_index_.dtype).itemsize
            doc_bin_.seek(pt_byte_offset)
            output_seq.append(np.frombuffer(doc_bin_.read(item_length),
                                            dtype=np.dtype(doc_index_.dtype)))
        
        pt_byte_offset, size = doc_index_[doc_idx_[doc_index_l_]]
        item_length = (offset_l_ + 1) * np.dtype(doc_index_.dtype).itemsize
        doc_bin_.seek(pt_byte_offset)
        output_seq.append(np.frombuffer(doc_bin_.read(item_length),
                                        dtype=np.dtype(doc_index_.dtype)))
    return output_seq


def perturb_dataset(args):
    print(">>>>>>>>>>>>>>>>>>>>>>>>>")
    print("WARNING: Index sizes will be inconsistent with the actual document boundaries.")
    print("<<<<<<<<<<<<<<<<<<<<<<<<<")

    perturbation_dir = args.perturbation_dir
    perturbation_files = sorted([pfnm[:-len('.bin')] for pfnm in os.listdir(perturbation_dir) if pfnm.endswith('.bin')])
    if args.perturbation_include_filters is not None:
        print("> Filtering perturbation files")
        perturbation_files = [pfnm for pfnm in perturbation_files if any(incl_filter in pfnm for incl_filter in args.perturbation_include_filters)]
        print(f">> Including perturbations with: {args.perturbation_include_filters}")
        print(f">> Perturbations to be used: {perturbation_files}")

    assert args.max_train_samples % args.max_train_batches == 0, "Inferred batch size is not an integer"
    batch_sz = args.max_train_samples // args.max_train_batches
    print(f"> Inferred batch size = {batch_sz}")
    effective_inj_batches = int(args.max_train_batches * (args.injection_loc_end - args.injection_loc_start))
    effective_inj_samples = effective_inj_batches * batch_sz
    
    samples_to_insert = 0
    for perturbation_nm in perturbation_files:
        print(f"> Loading: {os.path.split(perturbation_nm)}")
        mmap_perturb_set = MMapIndexedDataset(os.path.join(perturbation_dir, perturbation_nm))
        samples_to_insert += len(mmap_perturb_set)
    assert samples_to_insert <= args.max_train_samples
    print(f"> Inserting {samples_to_insert} samples.")
    print(f"> Perturbing {samples_to_insert / effective_inj_samples * 100}% of the effective training samples.")
    print(f"> Perturbing {samples_to_insert / effective_inj_batches * 100}% of the effective training batches (expectation).")

    doc_idx = np.load(f"{args.batch_info}_doc_idx.npy", allow_pickle=True, mmap_mode="r")
    sample_idx = np.load(f"{args.batch_info}_sample_idx.npy", allow_pickle=True, mmap_mode="r")
    shuffle_idx = np.load(f"{args.batch_info}_shuffle_idx.npy", allow_pickle=True, mmap_mode="r")

    rng = np.random.default_rng(args.seed)
    if args.loc_sampler == "seq":
        perturbation_locs = rng.choice(
            np.arange(int(np.floor(args.max_train_batches * args.injection_loc_start))*batch_sz, int(np.ceil(args.max_train_batches * args.injection_loc_end))*batch_sz),
            samples_to_insert, replace=False)
    elif args.loc_sampler == "batch":
        assert samples_to_insert < args.max_train_batches, f"More perturbations ({samples_to_insert}) than batches ({args.max_train_batches})"
        perturbation_batches = rng.choice(np.arange(int(np.floor(args.max_train_batches * args.injection_loc_start)), int(np.ceil(args.max_train_batches * args.injection_loc_end))),
                                          samples_to_insert, replace=False)
        perturbation_offsets = rng.choice(batch_sz, samples_to_insert, replace=True)
        perturbation_locs = perturbation_batches * batch_sz + perturbation_offsets
    else:
        raise NotImplementedError(f"Unknown loc_sampler: {args.loc_sampler}")

    assert perturbation_locs.max() < int(np.ceil(args.max_train_batches * args.injection_loc_end))*batch_sz, f"Perturbation locs max {perturbation_locs.max()} >= max allowed train samples {int(np.ceil(args.max_train_batches * args.injection_loc_end))*batch_sz}"
    assert perturbation_locs.min() >= int(np.floor(args.max_train_batches * args.injection_loc_start))*batch_sz, f"Perturbation locs min {perturbation_locs.min()} < min allowed train samples {int(np.floor(args.max_train_batches * args.injection_loc_start))*batch_sz}"
    assert perturbation_locs.max() < args.max_train_samples
    assert len(set(perturbation_locs.flatten())) == samples_to_insert
    print(f"> Perturbation locations batch range: {perturbation_locs.min()//batch_sz/args.max_train_batches:0.2f} to {perturbation_locs.max()//batch_sz/args.max_train_batches:0.2f}")

    doc_pointer = open(f"{args.raw_dataset}.bin", 'r+b')
    doc_index = MMapIndexedDataset.Index(f"{args.raw_dataset}.idx")

    p_ctr = 0
    prior_perturbation_locs = set()
    perturbation_info = []
    check_used = 0
    printed_ctr = 0
    for perturbation_nm in perturbation_files:
        print(f'> Begin adding file {perturbation_nm}')
        mmap_perturb_set = MMapIndexedDataset(os.path.join(perturbation_dir, perturbation_nm))
        assert max(mmap_perturb_set._index.sizes) <= args.train_seq_len

        for idx_ in trange(len(mmap_perturb_set)):
            one_pt_ex = mmap_perturb_set.get(idx_)

            pt_loc = perturbation_locs[p_ctr]
            pt_shuffle_idx = shuffle_idx[pt_loc]
            doc_index_f = sample_idx[pt_shuffle_idx][0]
            doc_index_l = sample_idx[pt_shuffle_idx + 1][0]
            offset_f = sample_idx[pt_shuffle_idx][1]
            offset_l = sample_idx[pt_shuffle_idx + 1][1]

            assert pt_loc not in prior_perturbation_locs
            prior_perturbation_locs.add(pt_loc)
            injection_details = {
                "perturbation_file": perturbation_nm,
                "perturbation_idx": idx_,
                "pt_loc": int(pt_loc),
                "pt_shuffle_idx": int(pt_shuffle_idx),
                "doc_index_f": int(doc_index_f), "doc_index_l": int(doc_index_l),
                "offset_f": int(offset_f), "offset_l": int(offset_l)
            }
            if args.injection_loc == "seq_shuffle":
                """
                Step 1: Sample an injection position and offset for the new training sequence window
                Step 2: Read in the training sequence
                Step 3: Create the  full sequence
                Step 4: Shorten the sequence to the train_seq_len
                Step 5: Inject it into the tokenized corpus
                """
                # Step 1: Sample an injection position and offset for the new training sequence window
                injection_pos = rng.choice(doc_index_l - doc_index_f + 1, 1)[0]
                if injection_pos == 0:
                    window_offset = 0
                else:
                    window_offset = rng.choice(len(one_pt_ex), 1)[0]
                # Step 2: Read in the orig training sequence
                pt_train_seqs = read_bin_file(doc_pointer, doc_index, doc_idx, doc_index_f, doc_index_l, offset_f, offset_l)
                pt_train_szs = [len(one_seq) for one_seq in pt_train_seqs]
                assert len(np.concatenate(pt_train_seqs)) == args.train_seq_len + 1
                # Step 3: Create the full sequence
                pt_train_seqs.insert(injection_pos, one_pt_ex)
                concat_pt_seq = np.concatenate(pt_train_seqs)
                # Step 4: Shorten the sequence to the train_seq_len
                if injection_pos > 0:
                    injection_start = sum(pt_train_szs[:injection_pos])
                    injection_end = injection_start + len(one_pt_ex)
                    used_check = False
                    if injection_start < window_offset:
                        used_check = True
                        print(f"> [DEBUG] Window check: Restricting window_offset ({window_offset}) <= injection_start ({injection_start}).")
                        window_offset = min(window_offset, injection_start)
                    if (window_offset + args.train_seq_len + 1) < injection_end:
                        used_check = True
                        print(f"> [DEBUG] Window check: Restricting window_offset ({window_offset}) + train_seq_len ({args.train_seq_len + 1}) >= injection_end ({injection_end}).")
                        window_offset = max(window_offset, injection_end - args.train_seq_len - 1)
                    if used_check:
                        check_used += 1
                else:
                    injection_start = 0
                
                injection_details["pt_injection_pos"] = int(injection_pos)
                injection_details["pt_window_offset"] = int(window_offset)
                injection_details["pt_injection_len"] = len(one_pt_ex)
                injection_details["orig_doc_seq_sizes"] = pt_train_szs

                concat_pt_seq = concat_pt_seq[window_offset:window_offset + args.train_seq_len + 1]
                assert (concat_pt_seq[injection_start - window_offset: injection_start - window_offset + len(one_pt_ex)] == one_pt_ex).all()
                # Step 5: Replace perturbation object with the full sequence and 
                # inject it into the tokenized corpus
                one_pt_ex = concat_pt_seq
                assert len(one_pt_ex) == args.train_seq_len + 1
            else:
                # The injected data will overwrite the existing training sequence
                # Overwtiting will happen at the start of the sequence
                pass

            perturbation_info.append(injection_details)

            if doc_index_f == doc_index_l:
                pt_byte_offset, _ = doc_index[doc_idx[doc_index_f]]
                pt_byte_offset += offset_f * np.dtype(doc_index.dtype).itemsize
                assert (offset_l - offset_f + 1) >= len(one_pt_ex)
                print(f">> Inserting perturbation {p_ctr} (len(perturbation) = {len(one_pt_ex)})")
                # print(f'>>> Inserting {len(one_pt_ex)} tokens starting at position {pt_byte_offset}')
                if not args.dry_run:
                    doc_pointer.seek(pt_byte_offset)
                    doc_pointer.write(one_pt_ex.tobytes(order="C"))
            else:
                pt_byte_offset, size = doc_index[doc_idx[doc_index_f]]
                pt_byte_offset += offset_f * np.dtype(doc_index.dtype).itemsize
                ex_space = size - offset_f
                ex_left = len(one_pt_ex) - ex_space
                print(f">> Inserting perturbation {p_ctr} (len(perturbation) = {len(one_pt_ex)})")
                # print(f'>>> Inserting {min(len(one_pt_ex), ex_space)} tokens starting at position {pt_byte_offset}')
                if not args.dry_run:
                    doc_pointer.seek(pt_byte_offset)
                    doc_pointer.write(one_pt_ex[:ex_space].tobytes(order="C"))
                ex_done = ex_space

                if ex_left > 0:
                    for i in range(doc_index_f + 1, doc_index_l):
                        pt_byte_offset, size = doc_index[doc_idx[i]]
                        ex_space = size
                        ex_left = ex_left - ex_space
                        # print(f'>>> Inserting {min(ex_left + ex_space, ex_space)} tokens starting at position {pt_byte_offset}')
                        if not args.dry_run:
                            doc_pointer.seek(pt_byte_offset)
                            doc_pointer.write(one_pt_ex[ex_done:ex_done+ex_space].tobytes(order="C"))
                        ex_done += ex_space
                        if ex_left <= 0:
                            break
                
                if ex_left > 0:
                    pt_byte_offset, size = doc_index[doc_idx[doc_index_l]]
                    ex_space = offset_l + 1
                    ex_left = ex_left - ex_space
                    # print(f'>>> Inserting {min(ex_left + ex_space, ex_space)} tokens starting at position {pt_byte_offset}')
                    if not args.dry_run:
                        doc_pointer.seek(pt_byte_offset)
                        doc_pointer.write(one_pt_ex[ex_done:ex_done+ex_space].tobytes(order="C"))
                    ex_done += ex_space

                assert ex_left <= 0
                assert ex_done >= len(one_pt_ex)
                printed_ctr += 1
        
            p_ctr += 1
            if args.dry_run and printed_ctr > 10:
                break
    
    with open(f"{args.raw_dataset}_perturbation_info.json", 'w') as fout:
        out_str = '\n'.join([json.dumps(one_info) for one_info in perturbation_info]) + '\n'
        fout.write(out_str)

    print(f"> [DEBUG] Window check used {check_used} times.")
    print("> Closing tokens file.")
    doc_pointer.close()


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--exp_name',
        required=True,
        help="The name of the experiment that will be run. For logging purposes only."
    )

    parser.add_argument(
        '--raw_dataset',
        required=True,
        help="the path to the tokenized base dataset"
    )

    parser.add_argument(
        '--batch_info',
        required=True,
        help="the path to the shuffled and sampled batch data"
    )

    parser.add_argument(
        '--perturbation_dir',
        required=True,
        help="the path to the perturbation data"
    )

    parser.add_argument(
        '--perturbation_include_filters',
        default=None,
        nargs='+',
        help="the list of filename filters (substrings) to include in the perturbation process"
    )

    parser.add_argument(
        '--max_train_samples',
        required=True,
        type=int,
        help="Maximum number of samples that will be used for model training"
    )

    parser.add_argument(
        '--max_train_batches',
        required=True,
        type=int,
        help="Maximum number of batches that will be used for model training"
    )

    parser.add_argument(
        '--train_seq_len',
        type=int,
        required=True,
        # default=1024,
        help="Sample length of training sequences"
    )
    
    parser.add_argument(
        '--injection_loc',
        choices=["seq_start", "seq_shuffle"],
        default="seq_start",
        help="Whether to inject perturbation data at the "
             "(1) overwrite start of a randomly sampled sequence, "
             "(2) shuffled into a training sequence (insert into and resize old data instead of overwriting any data)"
    )

    # parser.add_argument(
    #     '--eos_tok_id',
    #     type=int,
    #     default=50279,
    #     help="Token ID of <endoftext>"
    # )

    parser.add_argument(
        '--loc_sampler',
        choices=["seq", "batch"],
        default="seq",
        help="Whether to randomly sample a sequence or batch for injection"
    )

    parser.add_argument(
        '--injection_loc_start',
        type=float,
        default=0.0,
        help="Start of the range of batches for sampling the injection location. Should be in [0.0, 1.0]."
    )

    parser.add_argument(
        '--injection_loc_end',
        type=float,
        default=1.0,
        help="End of the range of batches for sampling the injection location. Should be in [0.0, 1.0]."
    )

    parser.add_argument(
        '--seed',
        required=True,
        type=int,
        help="the seed to use"
    )

    parser.add_argument(
        '--dry_run',
        action='store_true',
        help="Only simulate addition of the perturbation data"
    )

    return parser.parse_args()

if __name__=="__main__":
    args_ = parse_args()
    print("> Args:", vars(args_))
    perturb_dataset(args_)
