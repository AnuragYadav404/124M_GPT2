# library imports:
import tiktoken
import torch
import torch.nn as nn
from torch.nn import functional as F

# some parameters for our model
block_size = 8 # block size is the length of tokens our model sees for predicting the next probable token
batch_size = 4 # this defines the batch size of examples we feed in the model for training/inference
n_embd = 64
learning_rate = 3e-4
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
vocab_size = tokenizer.n_vocab

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
    # print(start_idx)
    xb = torch.stack([data[idx:idx+block_size] for idx in start_idx])
    yb = torch.stack([data[idx+1:idx+1+block_size] for idx in start_idx])
    return xb, yb


# fairly enough, we can just use get_batch to get the splits

# now lets build the model
# model is a transformer

# the first step is iteration:
# let's build the simple form and then iterate

class GPT2Model(nn.Module):
    # the model needs to define the init, forward pass and a generate pass
    # why only forward pass and not a generate pass?
    # forward pass is for training, generate pass is for inference, we can have different logic for the two passes, so we will define them separately
    # as for backward pass, we can just use an optimizer and attach the parameters
    # we can then control the training loop of the model outside of the model, so we don't need to define a backward pass here  
    # advantage of defining the training loop outside of the model is that we can have more control over the training process, such as learning rate scheduling, gradient clipping, etc.


    def __init__(self):
        super().__init__()
        # here we will be initializing the components of the model
        self.token_emb_table = nn.Embedding(vocab_size, n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, xb, yb=None):
        """
        xb shape: (B,T)
        yb shape: (B,T)
        here we will be defining the forward computation pass
        here we get input data (xb, yb)
        we return the logits
        inputs here: 
            xb is of shape: (B, T)
            yb is of shape: (B, T)
        then we do some computation based off xb
        end results is of shape (B, T, C), which are the logits
        C being the probabilites of next token
        C shape is of vocab_size here
        outputs: logits, loss
        logits is the output of the model, which is the unnormalized probabilities of the next token
        so if we have a vocabulary of size V, then the logits will be of shape (batch_size, block_size, V)
        so for each token in the input sequence, we have a vector of logits for the next token in the vocabulary
        loss is the cross entropy loss between the logits and the target tokens (yb)
        """
        # to test end to end, let's build a simple embedding mapping
        xb = self.token_emb_table(xb)
        logits = self.lm_head(xb)
        # so now logits become: B,T,n_embd

        if(yb is None):
            loss=None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C) # C is n_embd
            yb = yb.view(B*T)
            # here logits are of shape: B*T, n_embd -> but we don't want n_embd, we want n_vocab size
            # using this we know what are the class probs by the model
            # yb here is, B*T, where each individual token is memeber of space of n_vocab
            # print(xb.shape, yb.shape, logits.shape)
            loss = F.cross_entropy(logits, yb)

        return logits, loss

        
    def generate(self, xb, max_new_tokens):
        """
        # shape of xb: (B, T)
        here we will be generating the next set of tokens
        logits are helpful in generate as well
        logits, loss = model(xb, yb)
        # based on the logit of the last token, we will approximate our next token
        # that is logits is B,T,C
        # logits[:, -1, :] -> this is of size (B, 1, C)
        # so for each batch we have a probability of next token
        # we can now simply take sample from here
        # what does sampling mean here?
        # we are provided with xb, and we need to generate the next sequences based on the probs
        # next probs are done via the forward calls
        # 
        
        """
        for _ in range(max_new_tokens):
            # we need to make sure that in forward pass we are passing only context-size length tokens
            xb_cond = xb[:, -block_size:]
            # print(xb_cond)
            logits, loss = self(xb_cond) # here loss is mostly useless
            # logits is the interesting part
            # shape of logits: (B, T, C)
            # we are only interested in the last token
            logits = logits[:, -1, :]
            # logits is now of shape: (B, C)
            # now we apply softmax
            probs = F.softmax(logits, dim=-1)
            # now we sample from these probability distributions
            xb_next = torch.multinomial(probs, num_samples=1) # this is of shape (B, 1)
            # now we append this to original xb
            xb = torch.cat((xb, xb_next), dim=1) # (B, T+1)

        return xb


model = GPT2Model()

# let's add a optimizer here as well
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
steps = 1000
for step in range(steps):
    optimizer.zero_grad()
    xb, yb = get_batch()
    logits, loss = model(xb, yb)
    if(step%10 == 0):
        print(loss)
    # now we calculate gradients
    loss.backward()
    # we now to update the params
    optimizer.step()
    # here loss is a numerical value
    # logits is of size (B,T,C)
    # why is logits important? -> for computing cross entropy loss
    # we don't need them explicitly here in training loop we are defining

# let's create a dummy starting example for generating
xb_gen = torch.tensor([[0], [9]])
generated_tensor = model.generate(xb_gen, 20)
print(generated_tensor.shape)
tensor_values = generated_tensor[0].tolist()

# we need to map these token id based on vocab
# decode function takes raw tensor values
print(tokenizer.decode(tensor_values))