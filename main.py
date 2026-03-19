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

@dataclass
class GPTConfig:
    block_size: int = 256
    batch_size: int = 32
    n_embd: int = 512
    learning_rate: float = 3e-4


block_size = 64 
batch_size = 32 
n_embd = 768
learning_rate = 1e-4
max_grad_norm = 1.0
debug_block_stats = False

with open('./dataset/input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

tokenizer = tiktoken.get_encoding('gpt2')

data = torch.tensor(tokenizer.encode(text), dtype=torch.long, device=device) # data goes to device
vocab_size = tokenizer.n_vocab


def get_batch(split=None):
    start_idx = torch.randint(len(data)-block_size-1, (batch_size,), device=device)
    # print(start_idx)
    xb = torch.stack([data[idx:idx+block_size] for idx in start_idx])
    yb = torch.stack([data[idx+1:idx+1+block_size] for idx in start_idx])
    xb, yb = xb.to(device), yb.to(device)
    return xb, yb


# class Head(nn.Module):
#     def __init__(self, head_size, n_embd):
#         super().__init__()
#         self.query = nn.Linear(n_embd, head_size)
#         self.key = nn.Linear(n_embd, head_size)
#         self.value = nn.Linear(n_embd, head_size)
#         self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

#     def forward(self, xb):
#         # xb: (B,T,C), C = n_embd
#         B,T,C = xb.shape
#         q = self.query(xb) # output is (B,T,head_size)
#         k = self.key(xb)


#         # todo: F.scaled_dot_product_attention
#         head_size = q.shape[2]
#         wei = q@k.transpose(-2, -1)*(head_size**-0.5)
#         wei = wei.masked_fill(self.tril[:T,:T]==0, float('-inf'))
#         wei = wei.softmax(dim=-1)

#         v = self.value(xb) # v is B , T, head_size
#         out = wei@v # so out will be B,T, head_size

#         return out


# class MultiAttentionHead(nn.Module):

#     def __init__(self, num_heads, head_size, n_embd):
#         super().__init__()
#         self.heads = nn.ModuleList([Head(head_size=head_size, n_embd=n_embd) for _ in range(num_heads)])
#         self.linear = nn.Linear(num_heads*head_size, n_embd)
#     def forward(self, xb):
#         out = torch.cat([h(xb) for h in self.heads], dim=-1)
#         out = self.linear(out)
#         return out # out is (B,T,n_embd)

    
class CasualSelfAttention(nn.Module):
    def __init__(self, num_heads, head_size, n_embd, block_size):
        super().__init__()

        self.c_attn = nn.Linear(n_embd, 3*n_embd)
        self.c_proj = nn.Linear(n_embd, n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1
        self.num_heads = num_heads
        self.head_size = head_size
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)).view(1,1,block_size,block_size))
        
    def forward(self,x):
        B,T,n_embd = x.shape
        qkv = self.c_attn(x) # so qkv is of shape [B,T,3*n_embd] 
        q, k, v = qkv.split(n_embd, dim=2) # now we split qkv into q, k, v each of shape [B,T,n_embd] 
        # now the dimension 2 needs to be split based on num_heads
        # (B,nh,T,hs)
        k = k.view(B,T,self.num_heads, self.head_size).transpose(1,2) # so k is now [B,num_heads,T, head_size]
        q = q.view(B,T,self.num_heads, self.head_size).transpose(1,2) # so q is now [B,num_heads,T, head_size]
        v = v.view(B,T,self.num_heads, self.head_size).transpose(1,2) # so v is now [B,num_heads,T, head_size]

        #need to implement the flash-attention, and also compare the results
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        
        out = out.transpose(1, 2).contiguous().view(B, T, self.num_heads * self.head_size) # so out is now (B,T,n_embd)
        
        # now we can do the attention
        # wei = q@k.transpose(-2, -1)*(self.head_size**-0.5) # wei is B,nh,T,T
        # wei = wei.masked_fill(self.tril[:,:,:T,:T]==0, float('-inf'))
        # wei = wei.softmax(dim=-1) # wei is (B,nh,T,T)

        # out = wei@v # (B,nh,T,T) @ (B,nh,T,hs) -> B,nh,T,hs
        # out = out.transpose(1,2).contiguous().view(B,T,self.num_heads*self.head_size) # so out is now (B,T,n_embd)
        
        
        out = self.c_proj(out) # so out is now (B,T,n_embd) 
        return out


class MLP(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.linear1 = nn.Linear(n_embd, 4*n_embd) 
        self.gelu = nn.GELU(approximate='tanh')
        self.linear2 = nn.Linear(4*n_embd, n_embd)
        self.linear2.NANOGPT_SCALE_INIT = 1
    
    def forward(self,x):
        out = self.linear1(x)
        out = self.gelu(out)
        out = self.linear2(out)
        return out


class Block(nn.Module):
    def __init__(self, n_embd, num_heads):
        super().__init__()
        self.head_size = n_embd//num_heads
        self.cs_attn = CasualSelfAttention(num_heads=num_heads, head_size=self.head_size, n_embd=n_embd, block_size=block_size)
        self.ln1 = nn.LayerNorm(n_embd)
        self.mlp = MLP(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
    def forward(self, x, return_stats=False):
        x = x + self.cs_attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class GPT2Model(nn.Module):

    def __init__(self):
        super().__init__()

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(vocab_size, n_embd),
            wpe = nn.Embedding(block_size, n_embd),
            h = nn.ModuleList([Block(n_embd=n_embd, num_heads=8) for _ in range(6)]),
            ln_f = nn.LayerNorm(n_embd)
        ))

        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
        # we can use same lm_head and token_emb_table because of weight tying
        self.transformer.wte.weight = self.lm_head.weight
        self.apply(self._init_weights)
        # self.apply(self._print_weight_statistics)


    def _init_weights(self, module):

        std = 0.02
        if isinstance(module, nn.Linear):  
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                std *= (2*len(self.transformer.h))**-0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias) # even the biases are init to zeros
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
        

    def forward(self, xb, yb=None):
        B,T = xb.shape
        tok_emb = self.transformer.wte(xb)
        pos = torch.arange(0, T, device=xb.device)
        pos_emb = self.transformer.wpe(pos)
        x = tok_emb + pos_emb
        if self.training and debug_block_stats:
            for block_idx, block in enumerate(self.transformer.h):
                x = block(x)
                print(x.std().item())
            print('========')
        else:
            for block in self.transformer.h:
                x = block(x)

        x = self.transformer.ln_f(x)

        logits = self.lm_head(x) # shape of logits is (B,T,vocab_size)

        if(yb is None):
            loss=None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C) # C is vocab_size
            yb = yb.view(B*T)
            loss = F.cross_entropy(logits, yb)

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


model = GPT2Model().to(device=device)
use_amp = (device == "mps")
# model = torch.compile(model) does not seem to work with mps, so we will not use it for now, but it can be used for cuda
# import sys;
# sys.exit()

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
steps = 100
# we also want to print the count of parameters in the model
total_params = sum(p.numel() for p in model.parameters())
print(f'Total parameters: {total_params}')
# we probably also want to count the number of token the model trains upon during this time, which is B*T*steps
total_tokens = batch_size*block_size*steps
print(f'Total tokens trained on: {total_tokens}')

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
        print(f'token throughput: {B*T/dt} tokens/ms, Loss: {loss.item()}')
    loss.backward()
    # we also want to do gradient clipping here, which we will do via norm grad clip:
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
    optimizer.step()


start_text = "I am a language model, "
encoded_text = tokenizer.encode(start_text)
start_sequence = torch.tensor(encoded_text, dtype=torch.long).to(device=device)
start_sequence = start_sequence.unsqueeze(0) # add batch dimension

generated_tensor = model.generate(start_sequence, 60)
tensor_values = generated_tensor[0].tolist()

# # we need to map these token id based on vocab
# # decode function takes raw tensor values
print(tokenizer.decode(tensor_values))

