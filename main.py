# library imports:
import tiktoken
import torch


# some parameters for our model
block_size = 8 # block size is the length of tokens our model sees for predicting the next probable token
batch_size = 4 # this defines the batch size of examples we feed in the model for training/inference


# first we need to open and read the ./dataset/input.txt as utf-8 text file, and store the content in a variable called data
with open('./dataset/input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# --- IGNORE COPILOT ---
# steps for training a language model:
# 1. read the data
# 2. tokenize the data (convert characters to integers)
# 3. create input and target sequences
# 4. create a model
# 5. train the model    
# 
# --- IGNORE COPILOT ---
# if we are using tiktoken, should data be first transfered to tensor before tokenization? or can we directly tokenize the data as string?
# based on computation optimization,


tokenizer = tiktoken.get_encoding('gpt2')
# here it would be better to also move the data to GPU, but that is for a different case
data = torch.tensor(tokenizer.encode(text), dtype=torch.long)

# next we need to do is to create the dataset into splits of x and y
# since we are predicting the next token in the data set, we need to offset by 1

def get_batch(split=None):
    # lets first build the most simple one:
    # here we get only a single xb, and yb set
    # we first need to know the length of our context: block_size # defined as 8
    # we need a starting index 
        # conditions for starting index: random
        # sample space: 
        # n is data length, say we n = 100
        # start idx, can be 0 ... 100-block_size-1, -1 is because xy, is idx+1
    # why do we need a split?
    # split so that we take data based on training data or validation data

    start_idx = torch.randint(len(data)-block_size-1, (batch_size,))
    print(start_idx)
    xb = torch.stack([data[idx:idx+block_size] for idx in start_idx])
    yb = torch.stack([data[idx+1:idx+1+block_size] for idx in start_idx])
    return xb, yb


# fairly enough, we can just use get_batch to get the splits

# now lets build the model
# model is a transformer

# the first step is iteration:
# let's build the simple form and then iterate