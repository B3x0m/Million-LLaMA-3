import torch
import os

from config import ModelArgs
from model import Transformer

import json
from transformers import AutoTokenizer

# ----------------------------------------------------------------
model_path = "./ckpts/"
model_name = "VLotus"
tokenizer_path = "./tokenizer_files"
input_str = "\n" # the classic line
device = "cpu"
load_ckpt = False # load from lastest checkpoint instead of model
exec(open('configurator.py').read())
# ----------------------------------------------------------------

with open(f'{model_name}.json', 'r') as f:
    params_dict = json.load(f)
params = ModelArgs(**params_dict)
params.device = device

tokenizer = AutoTokenizer.from_pretrained(tokenizer_path) # , add_eos_token=True
model = Llama3(params, tokenizer).to(device)

def extract_number(file_name):
    return int(file_name.split('-')[1].split('.')[0])
# load lastest checkponts
if load_ckpt:
    files = [f for f in os.listdir(model_path) if os.path.isfile(os.path.join(model_path, f))]
    ckpt_files = [f for f in files if f.startswith('ckpt-') and f.endswith('.pth')]
    max_file = max(ckpt_files, key=extract_number)
    model_name = os.path.join(model_path, max_file)
    model_name = model_name.replace(".pth", "")
    print(f"Load from checkpoint: {model_name}")
    checkpoint = torch.load(f"{model_name}.pth", map_location=torch.device(device))
    model.load_state_dict(checkpoint['model_state_dict'])
else:
    model.load_state_dict(torch.load(f"{model_name}.pth", map_location=torch.device(device)))
# print(sum(p.numel() for p in model.parameters())/1e3, 'K parameters')
model.eval()

prompt = f"""Instruction:  {input_str}\nInput: \nOutput: """

# print("Normal output: " + model.generate(input_str))
output = model.generate(
    input_str,
    max_gen_len = params.max_seq_len - len(input_str), # our model doesn't have a built-in <endoftext> token so we have to specify when to stop generating
    memory_saver_div = 8, # the largest value we'll allow our query sequence length to get. makes memory consumption linear with respect to sequence length
    temperature = 0.9, # this is the default value that Llama3's official code has set
    top_p = 0.7, # this is the default value that Llama3's official code has set
    top_k = 10, # meta's code doesn't actually implement top_k selection but i've added it anyways as an alternative
)
print("Special output: " + output)
