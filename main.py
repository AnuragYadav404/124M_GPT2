# library imports:
import tiktoken
import torch
import torch.nn as nn
from torch.nn import functional as F

# some parameters for our model
block_size = 16 # block size is the length of tokens our model sees for predicting the next probable token
batch_size = 8 # this defines the batch size of examples we feed in the model for training/inference
n_embd = 256
learning_rate = 3e-4
# first we need to open and read the ./dataset/input.txt as utf-8 text file, and store the content in a variable called data
with open('./dataset/input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

tokenizer = tiktoken.get_encoding('gpt2')

data = torch.tensor(tokenizer.encode(text), dtype=torch.long) # data goes to device
vocab_size = tokenizer.n_vocab



def get_batch(split=None):
    # need to create train, validation splits

    start_idx = torch.randint(len(data)-block_size-1, (batch_size,))
    # print(start_idx)
    xb = torch.stack([data[idx:idx+block_size] for idx in start_idx])
    yb = torch.stack([data[idx+1:idx+1+block_size] for idx in start_idx])
    return xb, yb


class Head(nn.Module):
    # in head we need to do 3 things:
    # Query, Key and Value
    # Query and key will under go: matmul -> mask -> softmax
    # the result will then go matmul with value
    # query represents for each token, what it queries for
    # key represents what key each token represents, in context for a query
    # so a cross between query and a key, will give us high affinity for those with existing high affinity
    # value is the corresponding value due to the affinity
    def __init__(self, head_size, n_embd):
        super().__init__()
        self.query = nn.Linear(n_embd, head_size)
        self.key = nn.Linear(n_embd, head_size)
        self.value = nn.Linear(n_embd, head_size)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    def forward(self, xb):
        # xb: (B,T,C), C = n_embd
        B,T,C = xb.shape
        q = self.query(xb) # output is (B,T,head_size)
        k = self.key(xb)
        

        # now we want to do a matmul of query and key
        # TxC, TxC
        # [1,2,3][1,2,3] (2,3) || [1,2,3][1,2,3] || (2,3) 
        # at the end there is two token only
        # so it would be 2x2
        # "
        """ key      query.  -> (2x3)
        t1 [1,2,3]  [-1,0,1]
        t2 [4,5,6]  [2,0,1]

        at the end i want a 2x2:
        key      query.  -> (2x3)
        t1 [1,2,3]  [-1,0,1] [2,0,1]
        t2 [4,5,6]  
        t1k*t1q   t1k*t2q
        t2k*t1q   t2k*t2q
        """
        # "
        # TxT
        # the result would be a 
        head_size = q.shape[2]
        wei = q@k.transpose(-2, -1)*(head_size**-0.5)
        wei = wei.masked_fill(self.tril[:T,:T]==0, float('-inf'))
        wei = wei.softmax(dim=-1)

        # now wei is of shape: B, T, T

        # v is B,T, head_size
        # so a token, is being represented in head_size value dimension
        # when we do a matmul,
        # so each token is emitting a value
        # and of the knowledge learned from query and key of the regressive sequence, they are mapped to each token's value dimension
        v = self.value(xb) # v is B , T, head_size
        out = wei@v # so out will be B,T, head_size

        return out


class MultiAttentionHead(nn.Module):
    # we concat a set attention heads
    # number of attention heads can be distributed based off head_size
    def __init__(self, num_heads, head_size, n_embd):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size=head_size, n_embd=n_embd) for _ in range(num_heads)])
        self.linear = nn.Linear(num_heads*head_size, n_embd)
    def forward(self, xb):
        out = torch.cat([h(xb) for h in self.heads], dim=-1)
        out = self.linear(out)
        return out # out is (B,T,n_embd)
        

class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.linear1 = nn.Linear(n_embd, 4*n_embd)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(4*n_embd, n_embd)
    
    def forward(self,x):
        out = self.linear1(x)
        out = self.relu(out)
        out = self.linear2(out)
        return out


class Block(nn.Module):
    def __init__(self, n_embd, num_heads):
        super().__init__()
        self.head_size = n_embd//num_heads
        self.multi_head_attention1 = MultiAttentionHead(num_heads=num_heads,head_size=self.head_size,n_embd=n_embd)
        self.multi_head_attention2 = MultiAttentionHead(num_heads=num_heads,head_size=self.head_size,n_embd=n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.ffwd = FeedForward(n_embd)
    def forward(self,x):
        # need to implement leyer normalization here  
        # example: x = x + self.multi_head_attention1(self.ln1(x))
        # here layer normalization is applied before multi ahead because it is more stable for training, and it is also used in the original gpt paper, but it can be applied after as well, but it is not as stable for training
        # in original gpt paper
        x = x + self.multi_head_attention1(self.ln1(x))
        x = x + self.multi_head_attention2(self.ln2(x))
        x = x + self.ffwd(x)
        return x




class GPT2Model(nn.Module):

    def __init__(self):
        super().__init__()
        self.token_emb_table = nn.Embedding(vocab_size, n_embd)
        self.pos_emb_table = nn.Embedding(block_size, n_embd)
        self.block_net = nn.Sequential(
            Block(n_embd=n_embd, num_heads=8),
            Block(n_embd=n_embd, num_heads=8),
            Block(n_embd=n_embd, num_heads=8),
            Block(n_embd=n_embd, num_heads=8),
            Block(n_embd=n_embd, num_heads=8),
            Block(n_embd=n_embd, num_heads=8),
            Block(n_embd=n_embd, num_heads=8),
            Block(n_embd=n_embd, num_heads=8),
            Block(n_embd=n_embd, num_heads=8),
            Block(n_embd=n_embd, num_heads=8),
            Block(n_embd=n_embd, num_heads=8),
            Block(n_embd=n_embd, num_heads=8),
            Block(n_embd=n_embd, num_heads=8),
            Block(n_embd=n_embd, num_heads=8),
            Block(n_embd=n_embd, num_heads=8),
            Block(n_embd=n_embd, num_heads=8),
            Block(n_embd=n_embd, num_heads=8),
            Block(n_embd=n_embd, num_heads=8),
            Block(n_embd=n_embd, num_heads=8),
            Block(n_embd=n_embd, num_heads=8),
            Block(n_embd=n_embd, num_heads=8),
            Block(n_embd=n_embd, num_heads=8),
            Block(n_embd=n_embd, num_heads=8),
            Block(n_embd=n_embd, num_heads=8),
            nn.LayerNorm(n_embd)
        )
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, xb, yb=None):
        B,T = xb.shape
        tok_emb = self.token_emb_table(xb)
        pos_emb = self.pos_emb_table(torch.arange(T))
        x = tok_emb + pos_emb
        x = self.block_net(x)

        logits = self.lm_head(x)

        if(yb is None):
            loss=None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C) # C is n_embd
            yb = yb.view(B*T)
            loss = F.cross_entropy(logits, yb)

        return logits, loss

        
    def generate(self, xb, max_new_tokens):
        for _ in range(max_new_tokens):

            xb_cond = xb[:, -block_size:]

            logits, loss = self(xb_cond) # here loss is mostly useless

            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)

            xb_next = torch.multinomial(probs, num_samples=1) 

            xb = torch.cat((xb, xb_next), dim=1) # (B, T+1)

        return xb


model = GPT2Model()

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
steps = 100
for step in range(steps):
    optimizer.zero_grad()
    xb, yb = get_batch()
    logits, loss = model(xb, yb)
    if(step%10 == 0):
        print(loss)
    loss.backward()
    optimizer.step()


xb_gen = torch.tensor([[0], [9]])
generated_tensor = model.generate(xb_gen, 20)
print(generated_tensor.shape)
tensor_values = generated_tensor[0].tolist()

# we need to map these token id based on vocab
# decode function takes raw tensor values
print(tokenizer.decode(tensor_values))