import torch


file_path = '/Users/ishananand/Desktop/Transformers/extracted_text.txt'


def get_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()

    return text



text = get_data(file_path)
# remove UTF characters from text which got generated while parsing it
text = text.replace('\x00', '').replace('\ue04e', '').replace('\ue053', '')
characters = sorted(list(set(text)))
vocab_size = len(characters)
print("All the Unique Characters in the Text Data ", characters, " and it's length is ", vocab_size)


# Mapping all the characters to Integers
stoi = {} # characters to Integers
itos = {} # integers to characters
for i, ch in enumerate(characters):
    stoi[ch] = i

for i, ch in enumerate(characters):
    itos[i] = ch

encode = lambda s: [stoi[each_char] for each_char in s]
decode = lambda l: ''.join(itos[each_int] for each_int in l)

print("Encoded version of Hello Love: ", encode("Hello Love "))
print("Decoded Version of the above text ", decode(encode("Hello Love")))


def encode_dataset(text):
    data = torch.tensor(encode(text))
    print("First 1000 characters of Data", data[:1000])
    print("Shape and Data Type of Data ", data.shape, data.dtype)

    # splitting the data into train and test data
    n = int(0.9*(len(data)))
    train_data = data[:n]
    test_data = data[n:]

    return train_data, test_data


train_data, test_data = encode_dataset(text)

print(train_data[:100], test_data[:100])

# we cannot feed the entire data directly into the Transformer
# so what we do is split up the data into chunks and feed it up on every Node

def data_loader(data):
    block_size = 8
    X = data[:block_size]
    Y = data[1:block_size+1]
    for t in range(block_size):
        context = X[:t+1]
        target = Y[t]
        print(f" When input is {context} the target is {target}")

data_loader(train_data[:100])


batch_size = 4
block_size = 8
torch.manual_seed(1335) 
# such that next time random generator doesn't generate
# different numbers which it generated the previous time
def get_batch(data_type):
    batch_size = 4
    block_size = 8
    data_value = train_data if data_type =="train" else test_data
    ix = torch.randint(len(data_value) - block_size, (batch_size, ))
    X = torch.stack([data_value[i:i+block_size] for i in ix])
    Y = torch.stack([data_value[i+1:i+block_size+1] for i in ix])

    return X, Y

Xb, Yb = get_batch("train")

print(Xb, Xb.shape)
print(Yb, Yb.shape)

print(" The Input and output is something like this: ")
for b in range(batch_size):
    for t in range(block_size):
        context = Xb[b, :t+1]
        target = Yb[b, t]
        print(f" When input is {context} the target is {target}")


