import torch
import torch.nn as nn
import torch.nn.functional as F

import os
import time
import json
import math
import pandas as pd
from matplotlib import pyplot as plt
from dataclasses import asdict

from transformers import AutoTokenizer, LlamaTokenizerFast
from tokenizers import ByteLevelBPETokenizer
from config import ModelArgs
from model import Transformer


# -------------------------------------------------------------------
# Args for 111M params model
device = "cpu"
dim: int = 832 # Llama 3 8b uses 4096
n_layers: int = 10 # Llama 3 8b uses 32
n_heads: int = 8 # Llama 3 8b uses 32
n_kv_heads: int = 2 # Llama 3 8b's uses 8
vocab_size: int = 20000 # Llama 3 uses tokenizer of length 128256
multiple_of: int = 256 # Llama 3 8b's uses 1024
ffn_dim_multiplier: float = None # Llama 3 8b's uses 1.3
norm_eps: float = 1e-5
rope_theta: float = 10000 # Llama 3 8b uses 500000
max_batch_size: int = 12
max_seq_len: int = 512
dropout_rate: float = 0.1

weight_decay = 0.1
max_iters = 1000
eval_interval = 20 # log interval
warmup_iters = 15  # usually from 1 to 10% of max_iters
lr_init = 1e-3
warmup_factor = 2e-4  # Warmup factor
lr_final = 1e-4  # Minimum learning rate
eval_iters=10 # evaluate times

fp16 = False
train_tokenizer = False
save_ckpts=True
detect_anomaly = False
model_path = "./ckpts"
model_name = "VLotus"
file_path = "dataset.txt"
tokenizer_path = "./tokenizer_files"
exec(open('configurator.py').read())
# -------------------------------------------------------------------

if train_tokenizer:
    print("Training tokenizer...")
    tk_tokenizer = ByteLevelBPETokenizer()
    tokenizer_path = "tokenizer_files"
    tk_tokenizer.train(files=file_path, vocab_size=20000, min_frequency=2, special_tokens=[
        "<unk>",
        "<s>",
        "</s>"
    ])
    if not os.path.isdir(tokenizer_path):
        os.mkdir(tokenizer_path)
    tk_tokenizer.save(tokenizer_path + "/tokenizer.json")
    llama_tokenizer = LlamaTokenizerFast(
            tokenizer_file="./tokenizer_files/tokenizer.json",
            unk_token="<unk>",
            unk_token_id=0,
            bos_token="<s>",
            bos_token_id=1,
            eos_token="</s>",
            eos_token_id=2,
            max_model_input_sizes=512
        )

    llama_tokenizer.save_pretrained("./tokenizer_files")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
else:
    print("Loading saved tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

params = ModelArgs(dim, n_layers, n_heads,
                   n_kv_heads, vocab_size,
                   multiple_of, ffn_dim_multiplier,
                   norm_eps, rope_theta,
                   max_batch_size, max_seq_len,
                   device, dropout_rate)

if os.path.isfile("dataset.pt"):
    print("Have preprocessed dataset, loading....")
    dataset = torch.load('dataset.pt', map_location=torch.device(device)).to(device)
else:
    print("Preprocessing dataset...")
    def read_file_in_chunks(file_path, chunk_size=1024):
        with open(file_path, 'r', encoding='utf-8') as file:
            while True:
                data = file.read(chunk_size)
                if not data:
                    break
                yield data
    all_data = []
    for chunk in read_file_in_chunks(file_path):
        encoded_chunk = tokenizer.encode(chunk)
        tensor_chunk = torch.tensor(encoded_chunk, dtype=torch.long).to(device)
        all_data.append(tensor_chunk)
    dataset = torch.cat(all_data).to(device)
    torch.save(dataset, f'{file_path}.pt')
print("Total data tokens: " + str(dataset.shape))

n = int(0.9*len(dataset)) # first 90% will be our training dataset, the rest for validation
train_data = dataset[:n]
val_data = dataset[n:]
def get_batch(split, batch_size):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - params.max_seq_len, (batch_size,))
    x = torch.stack([data[i:i+params.max_seq_len] for i in ix])
    y = torch.stack([data[i+1:i+params.max_seq_len+1] for i in ix])
    x, y = x.to(params.device), y.to(params.device)
    return x, y

def lr_lambda(current_iter):
    if current_iter < warmup_iters:
        return warmup_factor + (1 - warmup_factor) * current_iter / warmup_iters
    else:
        decay_iters = max_iters - warmup_iters
        cosine_decay = 0.5 * (1 + math.cos(math.pi * (current_iter - warmup_iters) / decay_iters))
        return max(cosine_decay, lr_final / lr_init)

model = Transformer(params, tokenizer).to(params.device)
print(params)
print(sum(p.numel() for p in model.parameters())/1e6, 'M parameters')

optimizer = torch.optim.AdamW(model.parameters(), lr=lr_init, weight_decay=weight_decay, eps=1e-5)
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_iters - warmup_iters)

@torch.no_grad()
def estimate_loss(model, batch_size, eval_iters=eval_iters):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split, batch_size)
            logits, loss = model(X, targets=Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

print("Training...")
all_losses = []
if not os.path.isdir(model_path):
    os.mkdir(model_path)
if detect_anomaly:
    torch.autograd.set_detect_anomaly(True)
if fp16:
    scaler = torch.cuda.amp.GradScaler()
start_time = time.time()
for iter in range(max_iters):
    xb, yb = get_batch('train', params.max_batch_size)    
    xb, yb = xb.to(device), yb.to(device)
    optimizer.zero_grad(set_to_none=True)
    if fp16:
        with torch.cuda.amp.autocast():
            logits, loss = model(xb, targets=yb)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    else:
        logits, loss = model(xb, targets=yb)
        loss.backward()
        optimizer.step()
    scheduler.step()

    if iter % eval_interval == 0 or iter == max_iters - 1:
        current_time = time.time()
        elapsed_time = current_time - start_time
        losses = estimate_loss(model, params.max_batch_size)
        current_lr = optimizer.param_groups[0]['lr']
        print(f"step {iter}: lr {current_lr:.5f}, train loss {losses['train']:.2f}, val loss {losses['val']:.2f}, time elapsed: {elapsed_time:.2f}s, ETA: {elapsed_time * (max_iters - iter)/eval_interval :.3f}")
        all_losses += [losses]
        if save_ckpts:
            torch.save({
                        'iters': iter,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': losses['val'],
                        }, f"{model_path}/ckpt-{iter}.pth")
            print(f"Saved iters {iter} checkpoint.")
        start_time = time.time()

print("Model trained successfully! Saving model...")
torch.save(model.state_dict(), f"{model_name}.pth")
params_dict = asdict(params)
del params_dict['device']
with open(f'{model_name}.json', 'w') as json_file:
    json.dump(params_dict, json_file, indent=4)
    
for loss_record in all_losses:
    loss_record['train'] = loss_record['train'].item()
    loss_record['val'] = loss_record['val'].item()
df_losses = pd.DataFrame(all_losses)
ax = df_losses.plot()
plt.title('Loss Over Time')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.savefig('loss.png')
#torch.autograd.set_detect_anomaly(False)
