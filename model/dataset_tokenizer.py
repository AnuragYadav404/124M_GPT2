import tiktoken
import torch

with open('./dataset/input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

tokenizer = tiktoken.get_encoding('gpt2')

data_tokens = torch.tensor(tokenizer.encode(text), dtype=torch.long) 

import numpy as np
np.array(data_tokens, dtype=np.int32).tofile("./dataset/shakespeare.bin")

"""
import numpy as np

data = np.memmap("shakespeare.bin", dtype=np.int32, mode="r")


import torch

def get_batch():
    ix = torch.randint(len(data) - 128, (batch_size,))
    
    x = torch.stack([torch.from_numpy(data[i:i+128]) for i in ix])
    y = torch.stack([torch.from_numpy(data[i+1:i+129]) for i in ix])
    
    return x, y
"""