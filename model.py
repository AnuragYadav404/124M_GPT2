# library imports:
from pathlib import Path

from dataclasses import dataclass
import tiktoken
import torch
import torch.nn as nn
from torch.nn import functional as F
import time
import numpy as np

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
    block_size: int = 256           # block_size is: sequence size, the number of tokens in a sequence
    batch_size: int = 16            # batch_size is: number of sequences we process in parallel
    n_embd: int = 256               # n_embd is attention blocks dimensions
    learning_rate: float = 3e-4     # learning_rate here is declared as a constant -> might want to update these
    max_grad_norm = 1.0
    num_heads = 8
    debug_block_stats = False

gpt_config = GPTConfig()

debug_block_stats = False


tokenizer = tiktoken.get_encoding('gpt2')

data = np.memmap("./dataset/shakespeare.bin", dtype=np.int32, mode="r")
gpt_config.vocab_size = tokenizer.n_vocab




class DataLoaderShakespeare():
    def __init__(self, batch_size, block_size, process_rank, num_processes):

        self.batch_size = batch_size
        self.block_size = block_size

        self.process_rank = process_rank
        self.num_processes = num_processes

        self.position_offset = 0

        self.read_data()

        self.reset_start()


    def read_data(self):
        dat = np.memmap("./dataset/shakespeare.bin", dtype=np.int32, mode="r")
        self.tokens = torch.tensor(dat, dtype=torch.long)


    def reset_start(self):
        self.current_position = self.batch_size*self.block_size*self.process_rank + self.position_offset # each initialized at [0, 32, 64, 96]

    def get_batch(self, split=None):
        # start_idx of self is now computed
        B, T = self.batch_size, self.block_size

        buf = self.tokens[self.current_position : self.current_position+B*T+1]

        xb = (buf[:-1]).view(B, T) # inputs
        yb = (buf[1:]).view(B, T) # targets

        # now we are going to do checks are re_initialize our current_pos
        self.current_position += B * T * self.num_processes # [0 -> 128, 32->160 and so on]
        # now it is possible that next batch of loading is not possible
        if(self.current_position + (B*T*self.num_processes + 1) > len(self.tokens)): # here we check if we can get the next entire batch or no, otherwise we loop[]
            # here we have a single training data, that we loop over, so, we can include an offset
            if(self.position_offset + B*T*self.num_processes + 1 > len(self.tokens)):
                self.position_offset = 0
            else:
                self.position_offset +=1  # so all the next training loops start at index say 1, there will be a case when offset becomes big enough, that we have to reset it
            # if offset + B*T*num_process + 1 > len(self.tokens): here we reset offset
            self.reset_start()

        return xb, yb



# def get_batch(split=None):
#     start_idx = torch.randint(len(data)-block_size-1, (batch_size,))
#     # print(start_idx)
#     xb = torch.stack([torch.from_numpy(data[idx:idx+block_size].copy()) for idx in start_idx])
#     yb = torch.stack([torch.from_numpy(data[idx+1:idx+1+block_size].copy()) for idx in start_idx])
#     xb, yb = xb.to(device), yb.to(device)
#     return xb, yb

from experimental_arch.model_definition import GPT2Model


model = GPT2Model(config=gpt_config).to(device=device)
use_amp = (device == "mps")
# model = torch.compile(model) does not seem to work with mps, so we will not use it for now, but it can be used for cuda
# import sys;
# sys.exit()

optimizer = torch.optim.AdamW(model.parameters(), lr=gpt_config.learning_rate)
steps = 100
# we also want to print the count of parameters in the model
total_params = sum(p.numel() for p in model.parameters())
print(f'Total parameters: {total_params}')
# we probably also want to count the number of token the model trains upon during this time, which is B*T*steps
total_tokens = gpt_config.batch_size*gpt_config.block_size*steps
print(f'Total tokens trained on: {total_tokens}')


dataloader = DataLoaderShakespeare(batch_size=gpt_config.batch_size, block_size=gpt_config.block_size, process_rank=0, num_processes=1)

# as of current mode configs, we exceed mem at batch size 128, we can instead use grad_accum with batch_size of 32
# accum steps of 4
n_accum_steps = 8

for step in range(500):

    t0=time.time()

    optimizer.zero_grad()

    loss_accum = 0
    
    for accum_step in range(n_accum_steps):

        xb, yb = dataloader.get_batch()
        xb, yb = xb.to(device), yb.to(device)

        B,T = xb.shape
        
        logits, loss = model(xb, yb)
        
        

        # we want to accumulate loss here, and then do backward only after n_accum_steps
        # so here we need to mean out the loss, because if we are doing n_accum_steps,
        # gradients will be registered n_accum_steps time
        # so loss here needs to be representative of the n_accum_steps we take
        # if we just use normal loss, the gradients at the end will be: correct_val * n_accum_steps
        # say in case of no accum, we had loss = 4: 16/4 (4 x, 4 loss for each, averaged over 4)
        # now we use 2 accum: so in losses become: [4, 4]: (8/2)[4x2, 4x2] = [8,8]
        # Now this [4,4] will flow to the gradients and accumulate to a total loss of 8, so we average out by n_accum = 2 => [4/2, 4/2]

        loss = loss/n_accum_steps

        loss_accum += loss.item()

        loss.backward()

        
    # import code; code.interact(local=locals())
    torch.mps.synchronize()

    t1=time.time()

    if(step%10 == 0):
        dt = (t1-t0)*1000
        print(f'token throughput: {B*T/dt} tokens/ms, Loss: {loss_accum}')

    
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

