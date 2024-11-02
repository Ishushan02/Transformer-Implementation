import torch
import torch.nn as nn
device = torch.device("cpu")


class SelfAttention(nn.Module):

    def __init__(self, embed_size, heads):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        '''The EMbedding Size should be integer and the heads should equally divide up the embed_size'''

        assert (self.embed_size == self.head_dim * heads), " Didnot divide the heads equally, give proper heads so it equally divides the embeddings"

        self.values = torch.nn.Linear(self.head_dim, self.head_dim, bias = False)
        self.keys = torch.nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.query = torch.nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = torch.nn.Linear(heads*self.head_dim, embed_size) # Fully Connected Output Later


    def forward(self, values, keys, query, mask):
         N = query.shape[0] # this is number of training examples
         value_length, key_length, query_length = values.shape[1], keys.shape[1], query.shape[1]
         
         # splitting the Values into each heads pieces
         values = values.reshape(N, value_length, self.heads, self.heads)
         keys = keys.reshape(N, key_length, self.heads, self.heads)
         query = query.reshape(N, query_length, self.heads, self.heads)

         values = self.values(values) 
         query = self.query(query)
         keys = self.keys(keys)

         energy =  torch.einsum("nqhd,nkhd->nhqk",[query, keys]) #query @ keys.transpose(-2, -1) # or else do this torch.einsum("nqhd,nkhd->nhqk",[query, keys]) 
         # "{n(batchsize), q(querylen), h(head), d(head dimension)},{n(batchsize), k(keylength), h(head), d(head dimension)}->nhqk" 
         # queries in shape : (N, query_length, heads, head_dim)
         # keys in shape : (N, key_len, heads, head_dim)
         # enery in shape : (N, heads, query_len, key_len)

         #adding mask
         if mask is not None:
            energy = energy.masked_fill(mask==0, float("-1e20")) # we have to actually set inf

         attention = torch.softmax(energy/(self.embed_size**(1/2)), dim=3) # normalizing over key_length
         out = torch.einsum("nhql, nlhd->nqhd",[attention, values])  #attention @ values # torch.einsum("nhql, nlhd->nqhd",[attention, values])
         # {Attention(N, heads, query_len, key_len), Values(N, valueLength, heads, head_dim)}->(N, querylength, heads, headdim)
         out = out.reshape(N, query_length, self.heads*self.head_dim) # concatenating them ()

         fc_out = self.fc_out(out)

         return fc_out # flatten last two dimensions
    


class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, droupout, forward_expansion):
        super(TransformerBlock, self).__init__()

        self.attention = SelfAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size) # Batch norm for averages over entire batch, layer norm averages over  each value of layer
        self.norm2 = nn.LayerNorm(embed_size)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion*embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion*embed_size, embed_size)
        )
        self.dropout = nn.Dropout(droupout)

    def forward(self, value, key, query, mask):
        attention = self.attention(value, key, query, mask)
        x = self.dropout(self.norm1(attention+query))
        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward+x))
        return out
    

class Encoder(nn.Module):
    def __init__(self, src_vocab_size, embed_size, num_layers, heads, device, forward_expansion, dropout, max_length):
        super(Encoder, self).__init__()
        # max_length is related to positional Embedding

        self.embed_size = embed_size
        self.device = device
        self.word_embedding = nn.Embedding(src_vocab_size, embed_size)
        self.positional_embedding = nn.Embedding(max_length, embed_size)
        self.layers = nn.ModuleList(
            [
                TransformerBlock(embed_size=embed_size, heads=heads, droupout=dropout, forward_expansion=forward_expansion)

                
            for _ in range(num_layers)]
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):#
        N, sequence_length = x.shape
        positions = torch.arange(0, sequence_length).expand(N, sequence_length).to(self.device)
        out = self.dropout(self.word_embedding(x) + self.positional_embedding(positions))
        #                 adding positional embedding

        for layer in self.layers:
            out = layer(out, out, out, mask) # (in encoder all the key, query, values are same initially)

        return out
    

class DecoderBlock(nn.Module):
    def __init__(self, embed_size, heads, forward_expansion, dropout, device):
        super(DecoderBlock, self).__init__()
        self.attention = SelfAttention(embed_size=embed_size, heads=heads)
        self.norm = nn.LayerNorm(embed_size)
        
        self.transformer_block = TransformerBlock(embed_size=embed_size, heads=heads, droupout=dropout, forward_expansion=forward_expansion)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, value, key, src_mask, target_mask):
        attention = self.attention(x, x, x, target_mask) # 
        query = self.dropout(self.norm(attention+x)) # skip connection is also used in this
        out = self.transformer_block( value, key, query, src_mask)
        return out

class Decoder(nn.Module):
    def __init__(self, trg_vocab_size, embed_size, num_layers, heads, forward_expansion, dropout, max_length, device) :
        super(Decoder, self).__init__( )
        self.device = device
        self.word_embedding = nn.Embedding(trg_vocab_size,  embed_size)
        self.position_embedding = nn.Embedding(max_length, embed_size)

        self.layers = nn.ModuleList([
            DecoderBlock(embed_size=embed_size, heads=heads, forward_expansion=forward_expansion, dropout=dropout, device=device)
            for _ in range(num_layers) 

        ])

        self.fc_out = nn.Linear(embed_size, trg_vocab_size) # last linear layer
        self.dropout = dropout

                      #   input_to_decoder, encoder_output(to merge in decoder), src_mask, trg_mask 
    def forward(self, x,  enc_out, src_mask, trg_mask):
        N, seq_length = x.shape
        positions = torch.arange(0, seq_length).expand(N, seq_length).to(self.device)
        x = self.dropout((self.word_embedding(x) + self.position_embedding(positions))) 

        for layer in self.layers:
            x = layer(x, enc_out, src_mask, trg_mask)

        out = self.fc_out(x)
        return out


class Transformer(nn.Module):
    def __init__(self, src_vocab_size, trg_vocab_size, src_pad_idx, tar_pad_idx, embed_size = 256, num_layers = 6, forward_expansion = 4,
                 heads = 8, dropout=0, device = device, max_length = 100):
        super(Transformer, self).__init__()

        self.encoder = Encoder(src_vocab_size,
                               embed_size,
                               num_layers,
                               heads,
                               device,
                               forward_expansion,
                               dropout=0.2,
                               max_length=max_length)
        
        self.decoder = Decoder(
            trg_vocab_size,
            embed_size,
            num_layers,
            heads,
            forward_expansion,
            dropout=0.2,
            max_length=max_length,
            device=device
        )

        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = tar_pad_idx

    def make_src_mask(self, src):
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        # N, 1, 1, src_len
        return src_mask.to(device)
    
    def make_tar_mask(self, triangle):
        N, trg_len = triangle.shape
        trg_mask = torch.tril(torch.ones(trg_len, trg_len)).expand(N, 1, trg_len, trg_len)
        return trg_mask.to(device)
    
    def forward(self, src, trg):
        src_mask = self.make_src_mask(src)
        tar_mask = self.make_tar_mask(trg)

        enc_src = self.encoder(src, src_mask)
        out = self.decoder(trg, enc_src, src_mask, tar_mask) 

        return out
    
     



         

        

if __name__ == "__main__":
    device = device
    x =torch.tensor([[1, 5, 6, 4, 3, 9, 5, 2, 0], [1, 8, 7, 3, 4, 5, 6, 7, 2]]).to(device)
    trg = torch.tensor([[1, 7, 4, 3, 5, 9, 2, 0], [1, 5, 6, 2, 4, 7, 6, 2]]).to(device)

    src_pad_idx = 0
    trg_pad_idx = 0
    src_vocab_size =10
    trg_vocab_size = 10
    model = Transformer(src_vocab_size, trg_vocab_size, src_pad_idx, trg_pad_idx).to(device)

    out = model(x, trg[:, :-1])
    print(out, out.shape)

         
         

             








    




 
         




        


