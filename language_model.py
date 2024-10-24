import torch
from torch import nn
from torch.nn import functional as fn
from prepare_data import get_data, get_batch, encode_dataset

torch.manual_seed(1335) 

class BigramLanguageModel(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()
        # each_token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets = None):
        # idx and targets are both (B, T) tensor of integers
        logits = self.token_embedding_table(idx) # (B, T, C) (batch, WindowSize, Vocab Size) (4, 8, 63)
        
        if(targets == None):
            loss = None
        else:
            # To calculate the Loss it requires (B*T, T) as inputs
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T) # or targets.view(-1)

            loss = fn.cross_entropy(logits, targets)
        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            logits, loss = self(idx)
            logits = logits[:, -1, :] # it becomes (B, C)
                            # its only taking last 1 character in consideration for prediction
            # apply softmax to get probabilities
            probs = fn.softmax(logits, dim=-1) #(B, C)
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)

            idx = torch.cat((idx, idx_next), dim = 1)# (B, T + 1)
        return idx

    

file_path = '/Users/ishananand/Desktop/Transformers/extracted_text.txt'
text = get_data(file_path)

text = get_data(file_path)
# remove UTF characters from text which got generated while parsing it
text = text.replace('\x00', '').replace('\ue04e', '').replace('\ue053', '')
characters = sorted(list(set(text)))
vocab_size = len(characters)

stoi = {} # characters to Integers
itos = {} # integers to characters
for i, ch in enumerate(characters):
    stoi[ch] = i

for i, ch in enumerate(characters):
    itos[i] = ch

encode = lambda s: [stoi[each_char] for each_char in s]
decode = lambda l: ''.join(itos[each_int] for each_int in l)

print("All the Unique Characters in the Text Data ", characters, " and it's length is ", vocab_size)




train_data, test_data = encode_dataset(text)

Xb, Yb = get_batch("train")

m = BigramLanguageModel(vocab_size)
out, loss = m(Xb, Yb)
print(out.shape)
print("Loss: ", loss)

print(" Prediction before training the Model: ", decode(m.generate(torch.zeros((24, 27), dtype= torch.long), max_new_tokens=100)[0].tolist()))


optimizer = torch.optim.AdamW(m.parameters(), lr = 1e-3)

batch_size = 32
for step in range(30000): # 100 iterations

    #sample a batch of data
    Xb, Yb = get_batch("train")

    #evaluate the loss
    logits, loss = m(Xb, Yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
print("Loss is ", loss.item())

print(f" Prediction After training the Model  ", decode(m.generate(torch.zeros((1, 1), dtype= torch.long), max_new_tokens=100)[0].tolist()))
