"""
chars = sorted(list(set(text)))
vocab_size = len(chars)

stoi = { ch:i for i,ch in enumerate(chars) }
# print(stoi)
itos = { i:ch for i,ch in enumerate(chars) }
# print(itos)
# what does decoder do?
# takes a number, outputs a character
def encode(s):
    return [stoi[c] for c in s]

def decode(l):
    return ''.join([itos[i] for i in l])
"""



"""
# vocab with characters only:

# get characters from the data:
data = torch.tensor(encode(text), dtype=torch.long)
# """
