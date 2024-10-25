import torch
from torch import nn
from torch.nn import functional as F
# from prepare_data import get_data

# Defining the variables again here 
# hyperparameters
batch_size = 16 # how many independent sequences will we process in parallel?
block_size = 32 # what is the maximum context length for predictions?
max_iters = 5000
eval_interval = 100
learning_rate = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 64
n_head = 4
n_layer = 4
dropout = 0.0


torch.manual_seed(1337)

def get_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()

    return text

file_path = '/Users/ishananand/Desktop/Transformers/extracted_text.txt'
text = get_data(file_path)

text = get_data(file_path)
# remove UTF characters from text which got generated while parsing it
text = text.replace('\x00', '').replace('\ue04e', '').replace('\ue053', '')

print(text[:500])


# here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)
# create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string


# Train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data)) # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]

# data loading
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


class BigramLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        # encoding positions as well
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        # because we have to go to logits(loss) we need a linear layer language_model head
        self.lm_head = nn.Linear(n_embd, vocab_size)

        

    def forward(self, idx, targets=None):
        B, T = idx.shape
        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        #position embedding
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) #(T, C)
        # concatenating x tok embedding and position embedding
        x = tok_emb + pos_emb # (B, T, C) now x holds the position of token as well as along with its enocdings
        logits = self.lm_head(tok_emb) # (B,T, Vocab Size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

        
        

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        

model = BigramLanguageModel()
m = model.to(device)