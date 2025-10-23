import json
import torch
from safetensors import safe_open
from transformers import AutoTokenizer

hf_weights = safe_open("/lustre/fs01/External/nairr/USC/ameya/HubbleSuite/models/1B_lr-6e-4_tokens-100B_model-perturbed/global_step48000_hf_trial_bf16/model.safetensors",
                       framework="pt", device='cpu')
neox_layer0_weights = torch.load("../../HubbleSuite/models/1B_lr-6e-4_tokens-100B_model-perturbed/global_step48000/layer_02-model_00-model_states.pt")

hf_weights.metadata()
hf_weights.keys()

# torch.abs(neox_layer0_weights['attention.query_key_value.weight'] - torch.cat([hf_model.layers[0].self_attn.q_proj.weight, hf_model.layers[0].self_attn.k_proj.weight, hf_model.layers[0].self_attn.v_proj.weight])).max()

tok = AutoTokenizer.from_pretrained("models/1B_lr-6e-4_tokens-100B_model-perturbed/global_step48000_hf_trial_bf16/")
# hf_samples = []
# with open("models/debug/hf/output_sample/__lustre__fs01__External__nairr__USC__ameya__HubbleSuite__models__1B_lr-6e-4_tokens-100B_model-perturbed__global_step48000_hf_trial_bf16/samples_winogrande_hubble_2025-03-07T01-06-53.516565.jsonl") as fin:
#     for line in fin:
#         hf_samples.append(json.loads(line))
# for sample in hf_samples:
#     for gen_args in sample["arguments"].values():
#         g_arg0 = "<endoftext>" if len(gen_args['arg_0'])==0 else gen_args['arg_0']
#         print(g_arg0 + gen_args["arg_1"])

r1_input_ids = torch.load("models/debug/neox/embedding_input_ids.pt")
h1_input_ids = torch.load("models/debug/hf/embedding_input_ids.pt")

r1_input_map = {}
for i_t_, t_ in enumerate(r1_input_ids.tolist()):
    if tuple(t_) not in r1_input_map:
        r1_input_map[tuple(t_)] = []
    r1_input_map[tuple(t_)].append(i_t_)

print(r1_input_map)

r1_subset = [0, 1, 2, 3, 4, 5, 6, 10, 11, 12]
for h_id, r_id in zip(h1_input_ids, r1_input_ids[r1_subset]):
    print(tok.decode(h_id))
    print(tok.decode(r_id))
    print('-----')

r1_layer0_x = torch.load("models/debug/neox/layer0_x.pt")
h1_layer0_x = torch.load("models/debug/hf/layer0_x.pt")
# h2_layer0_x = torch.load("models/debug/hf-neox-repo/layer0_x.pt")
torch.abs(r1_layer0_x.transpose(0, 1)[r1_subset] - h1_layer0_x).max()

r1_layer0_cos = torch.load("models/debug/neox/layer0_cos.pt")
h1_layer0_cos = torch.load("models/debug/hf/layer0_cos.pt")
torch.abs(r1_layer0_cos.squeeze() - h1_layer0_cos.squeeze()).max()

r1_layer0_sin = torch.load("models/debug/neox/layer0_sin.pt")
h1_layer0_sin = torch.load("models/debug/hf/layer0_sin.pt")
torch.abs(r1_layer0_sin.squeeze() - h1_layer0_sin.squeeze()).max()

r1_layer0_attn_input = torch.load("models/debug/neox/layer0_attn_input.pt")
h1_layer0_attn_input = torch.load("models/debug/hf/layer0_attn_input.pt")
torch.abs(r1_layer0_attn_input.transpose(0, 1)[r1_subset] - h1_layer0_attn_input).max()

recreated_h1_layer0_query_states = torch.matmul(h1_layer0_attn_input, hf_model.layers[0].self_attn.q_proj.weight.T)

r1_layer0_attn_output = torch.load("models/debug/neox/layer0_attn_output.pt")
h1_layer0_attn_output = torch.load("models/debug/hf/layer0_attn_output.pt")
torch.abs(r1_layer0_attn_output.transpose(0, 1)[r1_subset] - h1_layer0_attn_output).max()

r1_layer0_query_states = torch.load("models/debug/neox/layer0_query_states.pt")
h1_layer0_query_states = torch.load("models/debug/hf/layer0_query_states.pt")
torch.abs(r1_layer0_query_states.transpose(0, 1).transpose(1, 2)[r1_subset] - h1_layer0_query_states).max()

r1_layer0_key_states = torch.load("models/debug/neox/layer0_key_states.pt")
h1_layer0_key_states = torch.load("models/debug/hf/layer0_key_states.pt")
torch.abs(r1_layer0_key_states.transpose(0, 1).transpose(1, 2)[r1_subset] - h1_layer0_key_states).max()

r1_layer0_value_states = torch.load("models/debug/neox/layer0_value_states.pt")
h1_layer0_value_states = torch.load("models/debug/hf/layer0_value_states.pt")
torch.abs(r1_layer0_value_states.transpose(0, 1).transpose(1, 2)[r1_subset] - h1_layer0_value_states).max()

r1_layer0_post_rot_query_states = torch.load("models/debug/neox/layer0_post_rot_query_states.pt")
h1_layer0_post_rot_query_states = torch.load("models/debug/hf/layer0_post_rot_query_states.pt")
torch.abs(r1_layer0_post_rot_query_states.transpose(0, 1).transpose(1, 2)[r1_subset] - h1_layer0_post_rot_query_states).max()

r1_layer0_post_rot_key_states = torch.load("models/debug/neox/layer0_post_rot_key_states.pt")
h1_layer0_post_rot_key_states = torch.load("models/debug/hf/layer0_post_rot_key_states.pt")
torch.abs(r1_layer0_post_rot_key_states.transpose(0, 1).transpose(1, 2)[r1_subset] - h1_layer0_post_rot_key_states).max()
