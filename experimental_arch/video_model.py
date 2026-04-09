# library imports:
from pathlib import Path

from dataclasses import dataclass
import tiktoken
import torch
import torch.nn as nn
from torch.nn import functional as F
import time

device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'
elif torch.backends.mps.is_available():
    device = 'mps'  

torch.manual_seed(1337)
# torch.backends.cuda.manual_seed_all(1337)
torch.mps.manual_seed(1337)


with open('./dataset/input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

tokenizer = tiktoken.get_encoding('gpt2')

data = torch.tensor(tokenizer.encode(text), dtype=torch.long, device=device) # data goes to device
vocab_size = tokenizer.n_vocab

block_size: int = 64 # max sequence length
vocab_size: int = 50257 # number of tokens: 50,000 BPE merges + 256 bytes tokens + 1 <|endoftext|> token
n_layer: int = 6 # number of layers
n_head: int = 8 # number of heads
n_embd: int = 512 # 
learning_rate: float = 3e-4
batch_size: int = 32


def get_batch(split=None):
    start_idx = torch.randint(len(data)-block_size-1, (batch_size,), device=device)
    # print(start_idx)
    xb = torch.stack([data[idx:idx+block_size] for idx in start_idx])
    yb = torch.stack([data[idx+1:idx+1+block_size] for idx in start_idx])
    xb, yb = xb.to(device), yb.to(device)
    return xb, yb


class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1
        # regularization
        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        # nh is "number of heads", hs is "head size", and C (number of channels) = nh * hs
        # e.g. in GPT-2 (124M), n_head=12, hs=64, so nh*hs=C=768 channels in the Transformer
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True) # flash attention
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
        # output projection
        y = self.c_proj(y)
        return y

class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu    = nn.GELU(approximate='tanh')
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x

class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

@dataclass
class GPTConfig:
    block_size: int = 64 # max sequence length
    vocab_size: int = 50257 # number of tokens: 50,000 BPE merges + 256 bytes tokens + 1 <|endoftext|> token
    n_layer: int = 6 # number of layers
    n_head: int = 8 # number of heads
    n_embd: int = 512 # embedding dimension

class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # weight sharing scheme
        self.transformer.wte.weight = self.lm_head.weight

        # init params
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        # idx is of shape (B, T)
        B, T = idx.size()
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"
        # forward the token and posisition embeddings
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device) # shape (T)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (T, n_embd)
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (B, T, n_embd)
        x = tok_emb + pos_emb
        # forward the blocks of the transformer
        for block in self.transformer.h:
            x = block(x)
            print(x.std().item())
        # forward the final layernorm and the classifier
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x) # (B, T, vocab_size)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss
    
    def generate(self, xb, max_new_tokens):
        for _ in range(max_new_tokens):
            with torch.no_grad():
                xb_cond = xb[:, -block_size:]

                logits, _ = self(xb_cond) # here loss is mostly useless

                logits = logits[:, -1, :] # so logits is now (B,C) where C is vocab size

                probs = F.softmax(logits, dim=-1) # we calculate the probabilities for the last token only, so probs is (B,C)
                topk_probs, topk_indices = torch.topk(probs, k=50, dim=-1) # now we pick the top k tokens, so topk_probs is (B,k) and topk_indices is (B,k)

                ix = torch.multinomial(topk_probs, num_samples=1) # multinomial will sample from the top k probabilities, so ix is (B,1) where the value is the index of the token in the top k
                # so ix is the index of the token in the top k, we need to convert it to the index in the vocab
                ix = topk_indices.gather(-1, ix) # so now ix is (B,1) where the value is the index of the token in the vocab
                # what does gather do? it takes the topk_indices and gathers the values at the indices specified by ix, so we get the actual token index in the vocab
                xb = torch.cat((xb, ix), dim=1) # (B, T+1)
        return xb




model = GPT(GPTConfig).to(device=device)
use_amp = (device == "mps")
# model = torch.compile(model) does not seem to work with mps, so we will not use it for now, but it can be used for cuda
import sys;
# sys.exit()

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
steps = 100
# we also want to print the count of parameters in the model
total_params = sum(p.numel() for p in model.parameters())
print(f'Total parameters: {total_params}')
# we probably also want to count the number of token the model trains upon during this time, which is B*T*steps
# total_tokens = batch_size*block_size*steps
# print(f'Total tokens trained on: {total_tokens}')

for step in range(steps):
    t0=time.time()
    optimizer.zero_grad()
    xb, yb = get_batch()
    B,T = xb.shape
    
    logits, loss = model(xb, yb)
        
    # import code; code.interact(local=locals())
    torch.mps.synchronize()
    t1=time.time()
    if(step%10 == 0):
        dt = (t1-t0)*1000
        # print(dt)
        print(f'token throughput: {B*T/dt} tokens/ms')
        print('Loss: ',loss)
    loss.backward()
    optimizer.step()


start_text = "I am a language model, "
encoded_text = tokenizer.encode(start_text)
start_sequence = torch.tensor(encoded_text, dtype=torch.long).to(device=device)
print(start_sequence.shape) # here we need to make sure the shape is (1, start_sequence_length)
start_sequence = start_sequence.unsqueeze(0) # add batch dimension
print(start_sequence.shape) 

generated_tensor = model.generate(start_sequence, 60)
print(generated_tensor.shape)
tensor_values = generated_tensor[0].tolist()

# # we need to map these token id based on vocab
# # decode function takes raw tensor values
print(tokenizer.decode(tensor_values))
