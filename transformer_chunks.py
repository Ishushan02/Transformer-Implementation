import torch
from torch import nn
from torch.nn import functional as Fn
# from prepare_data import get_data, get_batch, encode_dataset

# The M<athematical Trick in Self Attention Model
# SO, the Token should Communicate only with it's previous tokens and not the future Tokens
# WHat we will do is let's say we are at 5th Token so, what we do is we average the vectors of it's previous one and the current one
# Hence that's why we do tril which truncates all future tokens to Zero excluding there involvement

torch.manual_seed(1335) 

B, T, C = 4, 8, 2

x = torch.rand(B, T, C)
print(x.shape)

# We want x[b,t] = mean(i<=t) x[b, i]
xbow = torch.zeros((B, T, C))
for b in range(B):
    for t in range(T):
        xprev = x[b, :t+1]
        xbow[b, t] = torch.mean(xprev, 0)


# Doing the Above Loop formulation with Functions instead
torch.manual_seed(42)
a = torch.tril(torch.ones(3, 3)) 
# tril function outputs lower triangular matrix 
b = torch.randint(0, 10, (3, 2)).float()
c = a @ b # a * b
# so if you see the output of c it will be sum of the elements till that index

print("a- ")

print(a)

print("b- ")
print(b)

print("c- ")
print(c)

print("Now see the averaged Output  ----------------------------")
torch.manual_seed(42)
# now we just have to do 
a = torch.tril(torch.ones(3, 3)) 
a = a / torch.sum(a, 1, keepdim=True)
b = torch.randint(0, 10, (3, 2)).float()
c = a @ b # a * b
# so if you see the output of c it will be sum of the elements till that index

print("a- ")
print(a)

print("b- ")
print(b)

print("c- ")
print(c)



print(" Now Playing with Softmax Function The actual Self Attention Block ----------------- ")
tril = torch.tril((torch.ones(T, T)))
print(tril)
wei = torch.zeros(T, T)
print(wei)

print("masked wei where 0 filling it with -infinity")
wei = wei.masked_fill(tril == 0, float('-inf'))
print(wei)

print("Softmax of Wei ")
wei = Fn.softmax(wei, dim=-1)
print(wei)

xbow3 = wei @ x

print(torch.allclose(xbow, xbow3)) # tells whether both the vectors are Tur or Not



# Single Head of a Self Attention Model
print("-----------Single head of a Self Attention Model--------")

B, T, C = 4, 8, 32

x = torch.rand(B, T, C)
head_size = 16
key = nn.Linear(C, head_size, bias= False)
query = nn.Linear(C, head_size, bias=False)
value = nn.Linear(C, head_size, bias=False)

key = key(x) #(B, T, 16)
query = query(x) # (B, T, 16)

wei = query @ key.transpose(-2, -1) # (B, T, 16) @ (B, 16, T) --> (B, T, T)

tril = torch.tril(torch.ones((T, T)))

wei = wei.masked_fill(tril==0, float('inf'))
wei = Fn.softmax(wei, dim= -1)

v = value(x)
out = wei @ v # Dot Product of elements aggregated instead of raw x

print(out.shape)


'''

- Attention is a communication mechanism. Can be seen as nodes in a directed
graph looking at each other and aggregating information with a weighted sum 
from all nodes that point to them, with data-dependent weights.

- There is no notion of space. Attention simply acts over a set of vectors.
    This is why we need to positionally encode tokens.
- Each example across batch dimension is of course processed completely independently
 and never "talk" to each other
- In an "encoder" attention block just delete the single line that does 
   masking with tril, allowing all tokens to communicate. This block here
   is called a "decoder" attention block because it has triangular masking,
   and is usually used in autoregressive settings, like language modeling.
- "self-attention" just means that the keys and values are produced from 
   the same source as queries. In "cross-attention", the queries still get
   produced from x, but the keys and values come from some other, external 
   source (e.g. an encoder module)
- "Scaled" attention additional divides wei by 1/sqrt(head_size). This makes 
   it so when input Q,K are unit variance, wei will be unit variance too and 
   Softmax will stay diffuse and not saturate too much. Illustration below

'''