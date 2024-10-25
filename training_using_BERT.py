import torch
from torch import nn
from torch.nn import functional as F
# from prepare_data import get_data

# Defining the variables again here 
# hyperparameters
batch_size = 32 # how many independent sequences will we process in parallel?
block_size = 128 # what is the maximum context length for predictions?
max_iters = 1000
eval_interval = 100
learning_rate = 3e-4
device = torch.device("mps")
eval_iters = 100
n_embd = 384
n_head = 6
n_layer = 6
dropout = 0.2
head_size = 16


torch.manual_seed(1337)


import torch
from torch import nn
from torch.nn import functional as F
from transformers import BertTokenizer, BertModel

# Defining the variables again here 
# hyperparameters
batch_size = 64 
block_size = 256 
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
device = torch.device("mps")
eval_iters = 200
n_head = 6
n_layer = 6
dropout = 0.2

torch.manual_seed(1337)

# Initialize BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased').to(device)

def encode_dataset(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=block_size)
    data = inputs['input_ids']
    n = int(0.9*(len(data)))
    train_data = data[:n]
    test_data = data[n:]

    return train_data, test_data

def get_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
    return text

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters).to(device)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# Train and test splits

# Data loading
def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y



def encode_dataset(text):
    data = torch.tensor(encode(text))
    # print("First 1000 characters of Data", data[:1000])
    # print("Shape and Data Type of Data ", data.shape, data.dtype)

    # splitting the data into train and test data
    n = int(0.9*(len(data)))
    train_data = data[:n]
    test_data = data[n:]

    return train_data, test_data


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


# Batch Normalization for Tunning
class LayerNorm1d: # (used to be BatchNorm1d)

  def __init__(self, dim, eps=1e-5, momentum=0.1):
    self.eps = eps
    self.gamma = torch.ones(dim)
    self.beta = torch.zeros(dim)

  def __call__(self, x):
    # calculate the forward pass
    xmean = x.mean(1, keepdim=True) # batch mean
    xvar = x.var(1, keepdim=True) # batch variance
    xhat = (x - xmean) / torch.sqrt(xvar + self.eps) # normalize to unit variance
    self.out = self.gamma * xhat + self.beta
    return self.out

  def parameters(self):
    return [self.gamma, self.beta]
  


class Head(nn.Module):
    """ one head of a Self Attention """

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias= False)
        self.query = nn.Linear(n_embd, head_size, bias= False)
        self.value = nn.Linear(n_embd, head_size, bias= False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x) 
        q = self.query(x)

        wei = q @ k.transpose(-2, -1) # (B, T, 16) @ (B, 16, T) --> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T]==0, float('inf'))
        wei = F.softmax(wei, dim= -1) * C**-0.5 # (B, T, T)

        v= self.value(x)
        out = wei @ v

        return out



# Creating Multiple head Attention (just concatenating all the single head attention)
class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention running in parallel """

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out
    

# If you see paper there will be a feed forward network
class FeedFoward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)
    

# Making a Block so as to reiterate the entire diagram multiples times in one go
class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embd, n_head):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x)) # residual connections
        x = x + self.ffwd(self.ln2(x)) # residual connections
        return x

class BigramLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.bert_model = bert_model
        self.lm_head = nn.Linear(bert_model.config.hidden_size, tokenizer.vocab_size)

    def forward(self, idx, targets=None):
        outputs = self.bert_model(idx)
        logits = self.lm_head(outputs.last_hidden_state)  # Use the last hidden states for logits

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

        # if targets is None:
        #     loss = None
        # else:
        #     B, T, C = logits.shape
        #     logits = logits.view(B*T, C)
        #     targets = targets.view(B*T)
        #     loss = F.cross_entropy(logits, targets)

        # return logits, loss
    

    def generate(self, idx, max_new_tokens):
        # idx is B, T array of indices in the current context
        for _ in range(max_new_tokens):
            #crop the last block tokens
            idx_cond = idx[:, -block_size:]
            #get the predictions
            logits, loss= self(idx_cond)

            #focusing on the last time stamp
            logits = logits[:, -1, :] # becomes B, C
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append the result into resultant vector
            idx = torch.cat((idx, idx_next), dim=1) #(B, T+1)

        return idx

        

model = BigramLanguageModel()
m = model.to(device)





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

torch.manual_seed(1337)
module = LayerNorm1d(100)
x = torch.randn(32, 100) # batch size 32 of 100-dimensional vectors
x = module(x)
x.shape


train_data, test_data = encode_dataset(text)


Xb, Yb = get_batch("train")

model = BigramLanguageModel()
m = model.to(device)
# print the number of parameters in the model
print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):

    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=1000)[0].tolist()))