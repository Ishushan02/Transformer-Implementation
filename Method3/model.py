import torch
import torch.nn as nn
import math

class InputEmbeddings(nn.Module):

    # (dimension of embedding model, vocabulory size)
    def __init__(self, d_model:int, vocab_size:int):
        super(InputEmbeddings, self).__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)

    
    def forward(self, x):
        # the factor of sqrt(dimension of model size) is given by the author in the paper, so implementing it
        # later we also divide it with our model to normalize it
        return self.embedding(x) * math.sqrt(self.d_model) 
    
     
class PositionalEmbeddings(nn.Module):

    """
    So, basically we add position of each word so as to know which word is placed where,
    We do it by adding a same size of vector of embeddings  which includes some value given by the formula {formula below}
    which helps in recognising that the word is positioned at this position

    formula 

    denominator = pow(10000, (2*i/d_model)) , i is the position of the word and d_model is size of that word embedding 

    for even placed words
        sine(position/denominator)

    for odd placed words
        cosine(position/denominator)

    Practically we use logarithmic form in the denom as to stabilize the calculations
    """

    # dimension of embedding model, maximum length of a sentence, dropout layer
    def __init__(self, d_model:int, seq_len:int, dropout:float):
        super(PositionalEmbeddings, self).__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = dropout

        # As the row will be total length of the sentence(containing each word), and col will be of embedding size
        # creating the dimension of the Positional Embeddings
        self.position_embedding = torch.zeros (seq_len, d_model )
        # position of word declared with sequence length size array  (seq_len, 1)
        position = torch.arange(0, seq_len-1, dtype=torch.float).unsqueeze(1)  
        denominator = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000)/d_model))
        # [for all rows, startting from 0 and taking 2 steps every time at all even places, 0, 2, 4, 6, .... 2n]
        self.position_embedding[:, 0::2] = torch.sin(position/denominator)
        # [for all rows, startting from 1 and taking 2 steps every time at all odd places, 1, 3, 5, 7, 9, .... 2n+1]
        self.position_embedding[:, 1::2] = torch.cos(position, denominator)
         
        # Adding Batch Dimension to the Positional Embedding ((seq_len, d_model) --> unsqueeze at 0 position will give out (1, seq_len, d_model))
        self.position_embedding = self.position_embedding.unsqueeze(0) 

        # we will store this positional embedding in the buffer of this module
        self.register_buffer("position_embedding", self.position_embedding)

    def forward(self, x):
        # adding positional encoding to every sentence
        x = x + (self.position_embedding[:, :x.shape[1], :]).requires_grad(False) 
        # position_embedding[select all batches, select embeddings up to curent sequence length(here for x), select all embedding dimensions]
        # the second term helps us to not to train the positional encoding as these positions are fixed 
        x = self.dropout(x)
        # just adding dropout randomly
        return x
         


class LayerNormalization(nn.Module):
    '''
    We do layer normalization so ass to reduce the calculations and decrease the computation
    Let's say in our dataset we have 3 sentence examples s1, s2, s3 and vector associated to each of them is v1, v2, v3
    we normalize each on the op them independently (mean1, std_devia1) for s1, (mean2, std_devia2) for s2, (mean3, std_devia3) for s3
    x[i] = x[i] - mean/sqrt(std_deviation **2 + epislon  ) 

    '''
    def __init__(self, epislon: float = 10 ** -6) :
        super(LayerNormalization, self).__init__()
        self.epislon = epislon

        # we add 2 extra parameters just to have the command to increase or decrease the amplitude (alpha, bias)
        # we are using parameters function to remind that the parameters alpha and bias are trainable
        self.alpha = nn.Parameter(torch.ones(1)) # multiplied
        self.bias = nn.Parameter(torch.zeros(1)) # added 
         
    def forward(self, x):
        # mean of last dimension i.e everything after the batch
        x_mean = x.mean(dim = -1, keepdim = True)
        x_std = x.std(dim = -1, keepdim = True)
        normalized_data = self.alpha * (x - x_mean) / (x_std + self.bias) + self.bias
        
        return normalized_data
    


class FeedForwardBlock(nn.Module):

    '''
    It is a Fully connected layer, used in both Encoder and Decoder
    '''

    def __init__(self, d_model:int, d_ff:int, dropout:float ):
        super(FeedForwardBlock, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff) # W1, B1
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model) # W2, B2 

    def forward(self, x):
        x = self.linear1(x)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)

        return x


class MultiHeadAttentionBlock(nn.Module):

    '''
    See the paper in your notebook for better pictorial representation of this flow,
    '''
    #  embedding_dimension, number of Heads, dropout value
    def __init__(self, d_model:int, h:int, dropout: float):
        super(MultiHeadAttentionBlock, self).__init__()
        self.d_model = d_model
        self.h = h
        self.dropout = nn.Dropout(dropout)
        assert d_model%h == 0, " The embedding dimension(d_model) should be divisible by head "
        self.d_k = d_model // h

        self.w_k = nn.Linear(d_model, d_model) # Wk
        self.w_q = nn.Linear(d_model, d_model) # Wq
        self.w_v = nn.Linear(d_model, d_model) # Wv

        self.w_o = nn.Linear(d_model, d_model) # Wo


    @staticmethod # what it does is we can call this method anywhere without initilizing the class MultiHeadAttentionBlock
    def attention(query, key, value, mask, dropout:nn.Dropout):
        d_k = query.shape[-1]
        # (N, head, seq_len, d_k) * (N, head, d_k, seq_len) --> (N(batch), head, seq_len, seq_len)
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k) # transposing last 2 dimension for matrix multiplication

        if mask is not None:
            attention_scores.masked_fill_(mask==0, -1e9) # it's soo large

        attention_scores = attention_scores.softmax(dim = -1) # (Batch, head, seq_len, seq_len)

        if dropout is not None:
            attention_scores = dropout(attention_scores)
 
                # each head attention matrix, attention score
        return (attention_scores @ value), attention_scores #sotmax(Q.K/sqrt(d_k)) * Value attention matrix, attentionscores(for visualization)
        


    # (Query, Key, Value, Mask)
    def forward(self, q, k, v, mask):
        # (w_q * Q)
        query = self.w_q(q) # (N, seq_len, d_model) * (N, d_model, d_model) = (N, seq_len, d_model)
        # (w_k * K)
        key = self.w_k(k)  # (N, seq_len, d_model) * (N, d_model, d_model) = (N, seq_len, d_model)
        # (w_v * V)
        value = self.w_v(v) # (N, seq_len, d_model) * (N, d_model, d_model) = (N, seq_len, d_model) 

        # now dividing Query, Key, Value matrix into h heads
        # (keeping the batch same, keeping the sentence same, changing it to heads, changing it to d_k) , see notes for proper view
        # (Batch, seq_len, d_model) --> (batch, seq_len, h, d_k) --Transposing 1, 2 Col ----> (batch, h, seq_len, d_k)
        # we are transposing (1, 2) column because the view is (N(batch), head, seq_len, d_k), because
        # we want each head to watch seq_len, d_k
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1, 2)

        # output_of_softmax, attention_scores
        x, self.attention_scores = MultiHeadAttentionBlock.attention(query, key, value, mask, self.dropout)

        # Transposing it back --> (N(batch), head, seq_len, d_k) --> (N(batch), seq_len, head, d_k) --> (N(batch), seq_len, d_model)
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.h * self.d_k)

        # (N, seq_len, d_model) * (N, d_model, d_model) --> (N, seq_len, d_model)
        return self.w_o(x) # x * w_o is the output multiheaded attention

class ResidualConnection(nn.Module):
    '''
    Adding the Generic Residual Block
    '''
    def __init__(self, dropout:float ) -> None:
        super(ResidualConnection, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization()

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x))) # PreNormalization (but in paper it is PostNormalization)
    

class EncoderBlock(nn.Module):
    '''
    We are actually building a Encoder Block which contains [ MultiHead --> Norm --> FeedForward --> Norm] --> because this unit is 
    replicated Nx number of times in the Encoder - According to the paper

    In Encoder the MultiHead Attention Block is called as self attention block because it is applied to the same input with 3
    different roles (Query, Key, Value)
    '''

    def __init__(self, self_attention_block:MultiHeadAttentionBlock, feed_forward_block:FeedForwardBlock, dropout:float):
        super(EncoderBlock, self).__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        # self.dropout = nn.Dropout(dropout)

        # we need 2 residual conections so creating them
        self.residual_connection = nn.ModuleList([ResidualConnection(dropout) for _ in range(2)])

    def forward(self, x, src_mask):
        # using first residual connection 
        x = self.residual_connection[0](x, lambda x:self.self_attention_block(x, x, x, src_mask))
        # using second residual connection 
        x = self.residual_connection[1](x, lambda x:self.feed_forward_block(x))

        return x

         

class Encoder(nn.Module):
    '''
    Combining the EncoderBlocks Nx time
    '''

    def __init__(self, layers:nn.ModuleList) :
        super(Encoder, self).__init__()
        self.layers = layers
        self.norm = LayerNormalization()

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
         
        return self.norm(x)
         

class DecoderBlock(nn.Module):
    '''                                                         ( Cross Attention)
    Decoder Block which contains selfAttention --> Norm --> (Outp from Encoder, and Norm) --> Norm --> Feed Forward --> Norm
    This all are connected via 3 residual blocks(skip connection)
    and this Decoder Block is replicated Nx number of times
    '''

    def __init__(self, self_attention_block: MultiHeadAttentionBlock, cross_attention_block:MultiHeadAttentionBlock,  feed_forward_block:FeedForwardBlock, dropout:float):
        super(DecoderBlock, self).__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block

        # Here we have 3 of them
        self.residual_connection = nn.ModuleList([ResidualConnection(dropout) for _ in range(3)])

    def forward(self, x, encoder_output, src_mask, tar_mask): # src_mask is mask of Encoder and tar_mask is mask of Decoder
        #   here decoder mask is used in the first block og multiHead attention
        x = self.residual_connection[0](x, lambda x: self.self_attention_block(x, x, x, tar_mask))
        #  here in cross attention query is coming from Decoder, the key and Value is coming from Encoder and Encoder mask is used
        x = self.residual_connection[1](x, lambda x: self.cross_attention_block(x, encoder_output, encoder_output, src_mask ))
        x = self.residual_connection[2](x, lambda x:self.feed_forward_block(x))

        return x
    

class Decoder(nn.Module):
    '''
    Combining the Decoder Block Nx times
    '''
    def __init__(self, layers:nn.ModuleList) :
        super(Decoder, self).__init__()
        self.layers = layers
        self.norm = LayerNormalization()

    def forward(self, x, encoder_output, src_mask, tar_mask):
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tar_mask)

        return self.norm(x)


class ProjectionLayer(nn.Module):
    '''
    Adding the last layer of Decoder which has a Linear Layer and then followed by the softmax
    It's Input will be seq_len, d_model (see notes the output of Multi Head Attention)
    '''

    def __init__(self, d_model:int, vocab_size:int):
        super(ProjectionLayer, self).__init__()
        self.projection = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        # (N(batch), seq_len, d_model) --> (N(batch), seq_len, vocab_size)
        return torch.log_softmax(self.projection(x), dim=-1) 
    

class Transformer(nn.Module):
    '''
    Attaching the Encoder and Decoder Block in the Transformer Block
    '''
    def __init__(self, encoder:Encoder, decoder:Decoder, src_embed:InputEmbeddings, tgt_embed:InputEmbeddings, src_pos_embed:PositionalEmbeddings, tgt_pos_embed:PositionalEmbeddings, projection_layer:ProjectionLayer) :
        super(Transformer, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.src_pos_embed = src_embed
        self.tgt_pos_embed = tgt_pos_embed
        self.projection_layer = projection_layer

    def encode(self, src, src_mask):
        src = self.src_embed(src)
        src = self.src_pos_embed(src)

        return self.encoder(src, src_mask)
    
    def decode(self, encoder_output, src_mask, tgt, tgt_mask):
        tgt = self.tgt_embed(tgt)
        tgt = self.tgt_pos_embed(tgt)
        return self.decoder(tgt, encoder_output, src_mask, tgt_mask)
    
    def project(self, x):
        return self.projection_layer(x)
    

#                                                                                                                                                              d_ff - Hidden layer
def build_transformer(src_vocab_size:int, tgt_vocab_size:int, src_seq_len:int, tgt_seq_len:int, d_model: int = 512, Nx:int = 6, h:int = 8, dropout:float = 0.1, d_ff: int = 2048) ->Transformer:

    src_embed = InputEmbeddings(d_model, src_vocab_size)
    tgt_embed = InputEmbeddings(d_model, tgt_vocab_size)
    
    # Creating Positional Encoding Layers
    src_pos_embed = PositionalEmbeddings(d_model, src_seq_len, dropout)
    tgt_pos_embed = PositionalEmbeddings(d_model, tgt_seq_len, dropout)


    # Create the Encoder Block
    encoder_blocks = []
    for _ in range(Nx):
        encoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        encoder_block = EncoderBlock(encoder_self_attention_block, feed_forward_block, dropout)
        encoder_blocks.append(encoder_block)
    
    # Create the Decoder Block
    decoder_blocks = []
    for _ in range(Nx):
        decoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        decoder_multi_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        decoder_block = DecoderBlock(decoder_self_attention_block, decoder_multi_attention_block, feed_forward_block, dropout)
        decoder_blocks.append(decoder_block)

    # Create the Encoder and Decoder
    encoder = Encoder(nn.ModuleList(encoder_blocks))
    decoder = Decoder(nn.ModuleList(decoder_blocks))

    #Creating the Projection Layer
    projection_layer = ProjectionLayer(d_model, tgt_vocab_size)

    # The Transformer
    transformer = Transformer(encoder, decoder, src_embed, tgt_embed, src_pos_embed, tgt_pos_embed, projection_layer)

    #Initialize the Parameters (Xaviers Parameter, we can initialize any of them)
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    
    return transformer



