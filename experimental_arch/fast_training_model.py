### --------------------- LIBRARY IMPORTS --------------------- ###

from pathlib import Path
from dataclasses import dataclass
import tiktoken
import torch
import torch.nn as nn
from torch.nn import functional as F
import time
import numpy as np
### ----------------------------------------------------------- ###


### --------------------- DECLARE DEVICE TYPE --------------------- ###
device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'
elif torch.backends.mps.is_available():
    device = 'mps'  

### --------------------------------------------------------------- ###


### --------------------- DECLARE SEED --------------------- ###

torch.manual_seed(1337)
# torch.backends.cuda.manual_seed_all(1337)
torch.mps.manual_seed(1337)

### -------------------------------------------------------- ###


### --------------------- DECLARE GPT CONFIG --------------------- ###

@dataclass
class GPTConfig:
    block_size: int = 128           # block_size is: sequence size, the number of tokens in a sequence
    batch_size: int = 32            # batch_size is: number of sequences we process in parallel
    n_embd: int = 512               # n_embd is attention blocks dimensions
    learning_rate: float = 3e-4     # learning_rate here is declared as a constant -> might want to update these
    max_grad_norm = 1.0
    num_heads = 8

gpt_config = GPTConfig()

### --------------------- --------------------- --------------------- ###

debug_block_stats = False


### --------------------- DATA READ AND TOKENIZATION --------------------- ###

tokenizer = tiktoken.get_encoding('gpt2')

data = np.memmap("./dataset/shakespeare.bin", dtype=np.int32, mode="r")
vocab_size = tokenizer.n_vocab

### --------------------- --------------------- --------------------- ###

### --------------------- SIMPLE GET BATCH DATA LOADER --------------------- ###

# here we need to go a little ahead in designing a dataloader class

# so we have to decide on the type of data loader
# for now we are using tiny shakespear, but this will run out quick
# a better choice is: fineweb
# but for that we need to download big shard files


# let's say we continue to use shakespear dataset
# things remain almost the same, but we now need to take in the account of n_gpus, and hence iteration steps

class DataLoaderShakespeare():
    def __init__(self, batch_size, block_size, process_rank, num_processes):

        self.batch_size = batch_size
        self.block_size = block_size

        self.process_rank = process_rank
        self.num_processes = num_processes

        self.position_offset = 0

        self.read_data()

        self.reset()

    def read_data(self):
        # dat = np.load("./dataset/shakespeare.bin")
        # dat = dat.astype(np.int32) # added after video
        # self.tokens = torch.tensor(dat, dtype=torch.long)
        dat = np.memmap("./dataset/shakespeare.bin", dtype=np.int32, mode="r")
        self.tokens = torch.tensor(dat, dtype=torch.long)


    def reset_start(self):
        self.current_pos = self.batch_size*self.block_size*self.process_rank + self.position_offset # each initialized at [0, 32, 64, 96]

    def get_batch(self, split=None):
        # start_idx of self is now computed
        B, T = self.batch_size, self.block_size

        buf = self.tokens[self.current_position : self.current_position+B*T+1]

        x = (buf[:-1]).view(B, T) # inputs
        y = (buf[1:]).view(B, T) # targets

        # now we are going to do checks are re_initialize our current_pos
        self.current_pos += B * T * self.num_processes # [0 -> 128, 32->160 and so on]
        # now it is possible that next batch of loading is not possible
        if(self.current_pos + (B*T*self.num_processes + 1) > len(self.tokens)): # here we check if we can get the next entire batch or no, otherwise we loop[]
            # here we have a single training data, that we loop over, so, we can include an offset
            if(self.position_offset + B*T*self.num_processes + 1 > len(self.tokens)):
                self.position_offset = 0
            else:
                self.position_offset +=1  # so all the next training loops start at index say 1, there will be a case when offset becomes big enough, that we have to reset it
            # if offset + B*T*num_process + 1 > len(self.tokens): here we reset offset
            self.reset_start()

        return xb, yb




# so get_batch needs to be modified to take in the account of DDP
# we have n_rank, and a world_size
# so we will have to modify the start_idx to take in the account of world_size, so that each process gets a different part of the data
# so for say first iteration, we 
# so let's say B = 4, and T = 8
# so for each GPU, starting point will be: B*T*(n_rank), [0, 32, 64, 96]
# for the next iter: new_pos: start_pos + (B*T*num_proc): 0 + (8*4*4)
# and we also probably want to check if the next batch is possible to load or not
# so let's say: self.current_pos + (B*T*num_proc) + 1 > len(tokens): reset

# OR, we can have a fuzzy and random check
# so for each process, we are only considered if it can fetch the next batch or no
# this only works if, we have a single shard of data, but if we had many, we would have to iterate over them sequentially
# i guess, this is where a dataloader class helps





def get_batch(block_size, batch_size, process_rank,  process_world_size, split=None):
    start_idx = torch.randint(len(data)-block_size-1, (batch_size,))
    # print(start_idx)
    xb = torch.stack([torch.from_numpy(data[idx:idx+block_size].copy()) for idx in start_idx])
    yb = torch.stack([torch.from_numpy(data[idx+1:idx+1+block_size].copy()) for idx in start_idx])
    xb, yb = xb.to(device), yb.to(device)
    return xb, yb


### --------------------- ---------------------- --------------------- ###



### --------------------- MODEL DEFINITION --------------------- ###


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
#         wei = q@k.transpose(-2, -1)*(head_size**-0.5) @ q@k would give us: 
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
        # self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)).view(1,1,block_size,block_size))
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


class Block(nn.Module):
    def __init__(self, n_embd, num_heads, block_size):
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

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(vocab_size, self.config.n_embd),
            # wpe = nn.Embedding(block_size, n_embd),
            h = nn.ModuleList([Block(n_embd=self.config.n_embd, num_heads=self.config.num_heads, block_size=self.config.block_size) for _ in range(16)]),
            ln_f = nn.LayerNorm(self.config.n_embd)
        ))

        self.lm_head = nn.Linear(self.config.n_embd, vocab_size, bias=False)
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
        x = tok_emb
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

        
# generation part remains same
    def generate(self, xb, max_new_tokens):
        for _ in range(max_new_tokens):
            with torch.no_grad():
                xb_cond = xb[:, -self.config.block_size:]

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



### --------------------- --------------------- --------------------- ###


### --------------------- LIBRARIES FOR DDP AND SETUP --------------------- ###


from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import os

# set up DDP (distributed data parallel).

# torchrun command sets the env variables RANK, LOCAL_RANK, and WORLD_SIZE

ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?

if ddp:
    # use of DDP atm demands CUDA, we set the device appropriately according to rank
    
    assert torch.cuda.is_available(), "need CUDA for DDP"

    init_process_group(backend='nccl') # this is required for torch.distributed, CUDA GPUs use 'nccl'
    
    ddp_rank = int(os.environ['RANK']) # RANK gives us the "rank" or "process_number" for the current process seing this code -> helps identify b/w proc

    ddp_local_rank = int(os.environ['LOCAL_RANK'])  # LOCAL_RANK: gives the rank of a process based off num of gpus on a single node, useful for multi-node GPU clusters

    ddp_world_size = int(os.environ['WORLD_SIZE']) # WORLD_SIZE: gives us the total no of process that will be running

    device = f'cuda:{ddp_local_rank}' # since there are many different devices available we need indexes when declaring device: cuda:0, cuda:1 etc

    torch.cuda.set_device(device) # we set the devices

    master_process = ddp_rank == 0 # this process will do logging, checkpointing etc. # here we set a boolean for the master process
else:
    # vanilla, non-DDP run
    ddp_rank = 0
    ddp_local_rank = 0
    ddp_world_size = 1
    master_process = True
    # attempt to autodetect device
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    print(f"using device: {device}")

### --------------------- --------------------- --------------------- ###


### --------------------- torch.compile --------------------- ###

model = GPT2Model(config=gpt_config)

# we are only running on cuda for now
model = torch.compile(model) # only possible on cuda for now

model.to(device=device)

### --------------------- --------------------- --------------------- ###

optimizer = torch.optim.AdamW(model.parameters(), lr=gpt_config.learning_rate)
steps = 100
# we also want to print the count of parameters in the model
total_params = sum(p.numel() for p in model.parameters())
print(f'Total parameters: {total_params}')
# we probably also want to count the number of token the model trains upon during this time, which is B*T*steps
total_tokens = gpt_config.batch_size*gpt_config.block_size*steps*ddp_world_size # we multiply by world size to get the total tokens trained on across all processes
print(f'Total tokens trained on: {total_tokens}')

for step in range(500):
    t0=time.time()
    optimizer.zero_grad()

    xb, yb = get_batch(block_size=gpt_config.block_size, batch_size=gpt_config.batch_size)
    xb, yb = xb.to(device), yb.to(device)
    
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
    torch.nn.utils.clip_grad_norm_(model.parameters(), gpt_config.max_grad_norm)
    optimizer.step()

# generation part remains same
start_text = "I am a language model, "
encoded_text = tokenizer.encode(start_text)
start_sequence = torch.tensor(encoded_text, dtype=torch.long).to(device=device)
start_sequence = start_sequence.unsqueeze(0) # add batch dimension

generated_tensor = model.generate(start_sequence, 60)
tensor_values = generated_tensor[0].tolist()

# # we need to map these token id based on vocab
# # decode function takes raw tensor values
print(tokenizer.decode(tensor_values))

