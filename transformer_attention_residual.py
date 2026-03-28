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


block_size = 128 
batch_size = 32 
n_embd = 512
learning_rate = 1e-4
max_grad_norm = 1.0

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


class CustomRoFormerSinusoidalPositionalEmbedding(nn.Embedding):
    def __init__(self, seq_len: int, n_embd: int):
        super().__init__(seq_len, n_embd) # so now self.weight is of size seq_len x n_embd and initialized. We need to overwrite them
        self.weight = self._init_weight(self.weight)

    @staticmethod
    def _init_weight(out: nn.Parameter):
        seq_len,n_embd = out.shape
        out.requires_grad = False 
        i = torch.arange(1, n_embd/2+1, dtype=float) # so i becomes [1...d/2]
        # we get a user warning here
        thetas = (1/(10000**(2*(i-1)/n_embd))) # thetas are of shape: d/2
        positions = torch.arange(seq_len).unsqueeze(1) # pos of shape seq_len
        m_theta = positions*thetas # shape: seq_len x d/2
        cos =(m_theta.cos()) # seq_len x d/2
        sin =(m_theta.sin()) # seq_len x d/2
        # out = torch.cat([sin, cos], dim=-1) # seq_len x d # incorrect, as out loses the nn.Parameter properties, so we will assign them separately
        out[:, :n_embd//2] = sin
        out[:, n_embd//2:] = cos
        out.detach_()
        return out
    @torch.no_grad()
    def forward(self, seq_len):
        positions = torch.arange(seq_len, device=self.weight.device)
        return super().forward(positions)




class CasualSelfAttention(nn.Module):
    def __init__(self, num_heads, head_size, n_embd, block_size):
        super().__init__()

        self.c_attn = nn.Linear(n_embd, 3*n_embd)
        self.c_proj = nn.Linear(n_embd, n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1
        self.num_heads = num_heads
        self.head_size = head_size
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)).view(1,1,block_size,block_size))
        self.embed_positions = CustomRoFormerSinusoidalPositionalEmbedding(seq_len=block_size, n_embd=head_size)

    def apply_rotary_positional_embedding(self, x, sinusoidal_pos):
        # original dimensions of x is: (B,nh,T,d)
        if sinusoidal_pos is None:
            return x
        sin, cos = sinusoidal_pos # sin and cos are of shape: (1,1,T,d/2)
        x1, x2 = x[..., 0::2], x[..., 1::2] # here x1 is say (B,nh,T,d/2)
        # print(x1.shape, x2.shape, sin.shape, cos.shape)
        return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1) # (B,nh,T, d/2 *2) => (B,nh,T,d) what size does this become?


        
    def forward(self,x):
        # print("X shape is: ",x.shape)
        B,T,n_embd = x.shape
        qkv = self.c_attn(x) # so qkv is of shape [B,T,3*n_embd] 
        q, k, v = qkv.split(n_embd, dim=2) # now we split qkv into q, k, v each of shape [B,T,n_embd] 
        # now the dimension 2 needs to be split based on num_heads
        # (B,nh,T,hs)
        k = k.view(B,T,self.num_heads, self.head_size).transpose(1,2) # so k is now [B,num_heads,T, head_size]
        q = q.view(B,T,self.num_heads, self.head_size).transpose(1,2) # so q is now [B,num_heads,T, head_size]
        v = v.view(B,T,self.num_heads, self.head_size).transpose(1,2) # so v is now [B,num_heads,T, head_size]

        # Apply RoPE positional embedding to q and k
        sinusodial_pos = self.embed_positions(T)[ None, None, :, : ].chunk(2, dim=-1) 
        q = self.apply_rotary_positional_embedding(q, sinusodial_pos)
        k = self.apply_rotary_positional_embedding(k, sinusodial_pos)   

        #need to implement the flash-attention, and also compare the results
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
        
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


def block_attn_res(partial_block, blocks, norm, proj):
    V = torch.stack(blocks + [partial_block]) # this becomes a N,B,T,D
    # next we need queries and keys
    K = norm(V) # normalize on the values of dimension D for each token
    # Now we want to project the token values from Dimension D to a single dimension
    logits = proj(K).squeeze(dim=-1) # logits become (N,B,T)
    softm = logits.softmax(dim=0) # so across each layer info for each token representation we want to softmax
    # so as to weigh how much each layer's token representation contributes
    out =( V * softm.unsqueeze(-1)).sum(dim=0)
    # print("Out shape is: ", out.shape)
    return out



class Block(nn.Module):
    def __init__(self, n_embd, num_heads, res_attn_layer_number, res_attn_block_size):
        super().__init__()
        self.head_size = n_embd//num_heads
        self.cs_attn = CasualSelfAttention(num_heads=num_heads, head_size=self.head_size, n_embd=n_embd, block_size=block_size)
        self.ln1 = nn.LayerNorm(n_embd)
        self.mlp = MLP(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.attn_res_norm = nn.RMSNorm(n_embd)
        self.attn_res_proj = nn.Linear(n_embd, 1)
        self.mlp_res_norm = nn.RMSNorm(n_embd)
        self.mlp_res_proj = nn.Linear(n_embd, 1)
        self.res_attn_layer_number = res_attn_layer_number
        self.res_attn_block_size = res_attn_block_size

    def forward(self, blocks, hidden_states):
        partial_block = hidden_states
        h = block_attn_res(partial_block, blocks, self.attn_res_norm, self.attn_res_proj)

        if self.res_attn_layer_number % (self.res_attn_block_size//2) == 0:
            blocks = blocks + [partial_block]        # freeze snapshot
            partial_block = torch.zeros_like(partial_block)

        attn_out = self.cs_attn(self.ln1(h))

        partial_block = partial_block + attn_out

        h = block_attn_res(partial_block, blocks, self.mlp_res_norm, self.mlp_res_proj)

        mlp_out = self.mlp(self.ln2(h))

        partial_block = partial_block + mlp_out

        return blocks, partial_block


class AttnResidualGPT(nn.Module):

    def __init__(self):
        super().__init__()

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(vocab_size, n_embd),
            # wpe = nn.Embedding(block_size, n_embd),
            h = nn.ModuleList([Block(n_embd=n_embd, num_heads=8, res_attn_layer_number=i, res_attn_block_size=4) for i in range(1,6+1)]),
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
        # pos = torch.arange(0, T, device=xb.device)
        # pos_emb = self.transformer.wpe(pos)

        partial_block = tok_emb
        blocks = []

        for block in self.transformer.h:
            blocks, partial_block = block( blocks, partial_block)

        out = self.transformer.ln_f(partial_block)

        logits = self.lm_head(out) # shape of logits is (B,T,vocab_size)

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

                # let's add temperature here: 
                # by adding temp, we can control the randomness of the output, higher temp means more random, lower temp means more deterministic
                # temperature = 0.8
                # logits = logits / temperature
                logits = logits[:, -1, :] / 0.4 # so logits is now (B,C) where C is vocab size

                topk_probs, topk_indices = torch.topk(logits, k=50, dim=-1) # now we pick the top k tokens, so topk_probs is (B,k) and topk_indices is (B,k)

                probs = F.softmax(topk_probs, dim=-1) # we calculate the probabilities for the last token only, so probs is (B,C)

                ix = torch.multinomial(topk_probs, num_samples=1) # multinomial will sample from the top k probabilities, so ix is (B,1) where the value is the index of the token in the top k
                # so ix is the index of the token in the top k, we need to convert it to the index in the vocab
                ix = topk_indices.gather(-1, ix) # so now ix is (B,1) where the value is the index of the token in the vocab
                # what does gather do? it takes the topk_indices and gathers the values at the indices specified by ix, so we get the actual token index in the vocab
                xb = torch.cat((xb, ix), dim=1) # (B, T+1)

        return xb
    


model = AttnResidualGPT().to(device=device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
steps = 100
# we also want to print the count of parameters in the model
total_params = sum(p.numel() for p in model.parameters())
print(f'Total parameters: {total_params}')
# we probably also want to count the number of token the model trains upon during this time, which is B*T*steps
total_tokens = batch_size*block_size*steps
print(f'Total tokens trained on: {total_tokens}')

for step in range(500):
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