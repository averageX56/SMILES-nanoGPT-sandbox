"""
Run inference over a whole sample (a file of prompts) with a trained model.

Unlike sample.py (which draws a few free-running samples from a single prompt),
this script reads a file where each line is one prompt / prefix, generates one or
more completions for every line, and optionally writes everything to a CSV.

Typical use for a SMILES sandbox: put a list of SMILES prefixes in a text file
(one per line) and let the model complete each of them.

Example:
    python inference.py \
        --out_dir=out-small \
        --input_file=prompts.txt \
        --output_file=generations.csv \
        --num_samples=5 \
        --max_new_tokens=100

An empty line in the input file is treated as "generate from scratch" (the model
is seeded with the newline token).
"""
import os
import csv
import sys
import pickle
from contextlib import nullcontext
import torch
from models import get_model

# -----------------------------------------------------------------------------
out_dir = 'out'          # directory that holds the trained checkpoint (ckpt.pt)
model = 'base'           # architecture file in models/ to fall back on if the checkpoint doesn't record one
input_file = ''          # REQUIRED: text file, one prompt / prefix per line (the sample to run inference on)
output_file = ''         # optional: path to write results as CSV. Empty => only print to stdout
num_samples = 1          # how many completions to generate per prompt
max_new_tokens = 100     # number of tokens generated after each prompt
temperature = 0.8        # 1.0 = no change, < 1.0 = less random, > 1.0 = more random
top_k = 200              # retain only the top_k most likely tokens (None to disable)
stop_on_newline = True   # cut each generation at the first newline after the prompt (one molecule per line)
seed = 1337
device = 'cuda'          # 'cpu', 'cuda', 'cuda:0', etc.
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
compile = False          # use PyTorch 2.0 to compile the model to be faster
exec(open('configurator.py').read()) # overrides from command line or config file
# -----------------------------------------------------------------------------

if not input_file:
    raise SystemExit(
        "Please pass --input_file=<path>, a text file with one prompt/prefix per line.\n"
        "Example: python inference.py --out_dir=out-small --input_file=prompts.txt"
    )

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
device_type = 'cuda' if 'cuda' in device else 'cpu'
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# load the checkpoint and rebuild the model it was trained with
ckpt_path = os.path.join(out_dir, 'ckpt.pt')
checkpoint = torch.load(ckpt_path, map_location=device)
model_name = checkpoint.get('config', {}).get('model', model)
ModelConfig, Model = get_model(model_name)
modelconf = ModelConfig(**checkpoint['model_args'])
net = Model(modelconf)
state_dict = checkpoint['model']
unwanted_prefix = '_orig_mod.'
for k, v in list(state_dict.items()):
    if k.startswith(unwanted_prefix):
        state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
net.load_state_dict(state_dict)
net.eval()
net.to(device)
if compile:
    net = torch.compile(net)

# load the tokenizer (stoi/itos) from the dataset's meta.pkl
meta_path = None
if 'config' in checkpoint and 'dataset' in checkpoint['config']:
    meta_path = os.path.join('data', checkpoint['config']['dataset'], 'meta.pkl')
if meta_path is None or not os.path.exists(meta_path):
    raise FileNotFoundError(
        "Could not find meta.pkl for this checkpoint's dataset. The sandbox needs the "
        "dataset's meta.pkl (with stoi/itos) to encode/decode. "
        f"Expected it at: {meta_path!r}"
    )
print(f"Loading meta from {meta_path}...")
with open(meta_path, 'rb') as f:
    meta = pickle.load(f)
stoi, itos = meta['stoi'], meta['itos']

def encode(s):
    return [stoi[c] for c in s]

def decode(l):
    return ''.join([itos[i] for i in l])

# read the sample: one prompt / prefix per line
with open(input_file, 'r', encoding='utf-8') as f:
    prompts = [line.rstrip('\n') for line in f]
print(f"Loaded {len(prompts)} prompt(s) from {input_file}")

results = []  # list of (prompt_index, prompt, sample_index, generation)
with torch.no_grad():
    with ctx:
        for p_idx, prompt in enumerate(prompts):
            seed_text = prompt if prompt != '' else '\n'  # empty line => sample from scratch
            try:
                start_ids = encode(seed_text)
            except KeyError as e:
                print(f"[skip] prompt {p_idx} contains a token not in the vocabulary: {e}", file=sys.stderr)
                continue
            # replicate the prompt across the batch dim to draw num_samples completions at once
            x = torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...].repeat(num_samples, 1)
            y = net.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
            for s_idx in range(num_samples):
                gen = decode(y[s_idx].tolist())
                if stop_on_newline:
                    nl = gen.find('\n', len(seed_text))
                    if nl != -1:
                        gen = gen[:nl]
                results.append((p_idx, prompt, s_idx, gen))
                print(f"[{p_idx}:{s_idx}] {gen}")

# optionally save everything to a CSV
if output_file:
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['prompt_index', 'prompt', 'sample_index', 'generation'])
        writer.writerows(results)
    print(f"\nWrote {len(results)} generation(s) to {output_file}")
