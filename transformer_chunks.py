import torch
from torch import nn
from torch.nn import functional as Fn
# from prepare_data import get_data, get_batch, encode_dataset

# The M<athematical Trick in Self Attention Model
# SO, the Token should Communicate only with it's previous tokens and not the future Tokens
# WHat we will do is let's say we are at 5th Token so, what we do is we average the vectors of it's previous one and the current one

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