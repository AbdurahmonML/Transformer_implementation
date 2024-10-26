import torch
from torch import nn
import torch.optim as optim
import math
import numpy as np

Norm = nn.LayerNorm

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()

        self.encoding = torch.zeros(max_len, d_model)

        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        self.encoding[:, 0::2] = torch.sin(position * div_term)

        self.encoding[:, 1::2] = torch.cos(position * div_term)
        self.encoding = self.encoding.unsqueeze(0)

    def forward(self, x):
        x = x + self.encoding[:, :x.size(1), :].to(x.device)
        return x

class SelfAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, dropout=0.1):
        super(SelfAttention, self).__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        key_dim = key.size(-1)
        attn = torch.matmul(query / np.sqrt(key_dim), key.transpose(2, 3))
        if mask is not None:
            mask = mask.unsqueeze(1)
            attn = attn.masked_fill(mask == 0, -1e9)
        attn = self.dropout(torch.softmax(attn, dim=-1))
        output = torch.matmul(attn, value)

        return output

# Multi-head attention layer
class MultiHeadAttention(nn.Module):
    def __init__(self, embedding_dim, num_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        self.embedding_dim = embedding_dim
        self.self_attention = SelfAttention(dropout)
        # The number of heads
        self.num_heads = num_heads
        # The dimension of each head
        self.dim_per_head = embedding_dim // num_heads
        # The linear projections
        self.query_projection = nn.Linear(embedding_dim, embedding_dim)
        self.key_projection = nn.Linear(embedding_dim, embedding_dim)
        self.value_projection = nn.Linear(embedding_dim, embedding_dim)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, query, key, value, mask=None):
        # Apply the linear projections
        batch_size = query.size(0)
        query = self.query_projection(query)
        key = self.key_projection(key)
        value = self.value_projection(value)
        # Reshape the input
        query = query.view(batch_size, -1, self.num_heads, self.dim_per_head).transpose(1, 2)
        key = key.view(batch_size, -1, self.num_heads, self.dim_per_head).transpose(1, 2)
        value = value.view(batch_size, -1, self.num_heads, self.dim_per_head).transpose(1, 2)
        # Calculate the attention
        scores = self.self_attention(query, key, value, mask)
        # Reshape the output
        output = scores.transpose(1, 2).contiguous().view(batch_size, -1, self.embedding_dim)
        # Apply the linear projection
        output = self.out(output)
        return output

class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, ff_dim = 2048, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()
        self.dropout = dropout
        self.nhead = nhead
        self.d_model = d_model


        self.attention = MultiHeadAttention(d_model, nhead)
        self.norm1 = Norm(d_model)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, d_model)
        )
        self.norm2 = Norm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        x2 = self.norm1(x)

        x = x + self.dropout1(self.attention(x2, x2, x2, mask))
        x2 = self.norm2(x)
        x = x + self.dropout2(self.feed_forward(x2))
        return x

# Transformer decoder layer
class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super(TransformerDecoderLayer, self).__init__()
        self.dropout = dropout
        self.d_model = d_model
        self.nhead = nhead


        self.self_attention = MultiHeadAttention(d_model, nhead, dropout)
        self.encoder_attention = MultiHeadAttention(d_model, nhead, dropout)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Linear(dim_feedforward, d_model)
        )
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.norm1 = Norm(d_model)
        self.norm2 = Norm(d_model)
        self.norm3 = Norm(d_model)

    def forward(self, x, memory, source_mask, target_mask):
        x2 = self.norm1(x)
        x = x + self.dropout1(self.self_attention(x2, x2, x2, target_mask))
        x2 = self.norm2(x)
        x = x + self.dropout2(self.encoder_attention(x2, memory, memory, source_mask))
        x2 = self.norm3(x)
        x = x + self.dropout3(self.feed_forward(x2))
        return x

class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_encoder_layers):
        super(TransformerEncoder, self).__init__()
        # Provide your implementation here...
        dropout = encoder_layer.dropout
        self.num_layers = num_encoder_layers
        self.num_heads = encoder_layer.nhead
        self.embedding_dim = encoder_layer.d_model
        self.layers = nn.ModuleList([TransformerEncoderLayer(self.embedding_dim, self.num_heads, 2048, dropout) for _ in range(num_encoder_layers)])
        self.norm = Norm(self.embedding_dim)
        self.pos_emb = PositionalEncoding(self.embedding_dim)

    def forward(
        self,
        src,
        mask=None,
        src_key_padding_mask=None
    ):
        # Provide your implementation here...
        #x = self.pos_emb(src)
        x = src
        for layer in self.layers:
            x = layer(x, mask)
        x = self.norm(x)

        return x

class TransformerDecoder(nn.Module):
    def __init__(
        self,
        decoder_layer,
        num_decoder_layers
    ):
        #decoder_layer(d_model, nhead, dim_feedforward, dropout)
        super(TransformerDecoder, self).__init__()
        # Provide your implementation here...
        embedding_dim = decoder_layer.d_model
        num_heads = decoder_layer.nhead
        dropout = decoder_layer.dropout
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.position_embedding = PositionalEncoding(embedding_dim)
        self.layers = nn.ModuleList([TransformerDecoderLayer(embedding_dim, num_heads, 2048, dropout) for _ in range(num_decoder_layers)])
        self.norm = Norm(embedding_dim)

    def forward(
        self,
        tgt,
        memory,
        tgt_mask=None,
        source_mask = None
    ):
        # Provide your implementation here...
        #x = self.embedding(tgt)
        #x = self.position_embedding(x)
        x = tgt
        for layer in self.layers:
            x = layer(x, memory, source_mask, tgt_mask)
        x = self.norm(x)
        return x

class TransformerModel(nn.Module):
    def __init__(
        self,
        src_vocab_size,        # Vocabulary size of the source language (input)
        tgt_vocab_size,        # Vocabulary size of the target language (output)
        d_model=512,           # Dimension of model embeddings (default 512)
        nhead=8,               # Number of attention heads in multi-head attention (default 8)
        num_encoder_layers=6,  # Number of layers in the Transformer encoder (default 6)
        num_decoder_layers=6,  # Number of layers in the Transformer decoder (default 6)
        dim_feedforward=2048,  # Dimension of the feedforward network inside the Transformer (default 2048)
        dropout=0.1            # Dropout rate (default 0.1)
    ):
        # Initialize the nn.Module parent class
        super(TransformerModel, self).__init__()

        # Source embedding layer that converts input tokens to embeddings of size d_model
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)

        # Target embedding layer that converts target tokens to embeddings of size d_model
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)

        # Positional encoding for the embeddings to encode the positions of tokens in the sequence
        self.positional_encoding = PositionalEncoding(d_model)
        # Create a single layer of Transformer encoder
        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
        # Stack multiple encoder layers (num_encoder_layers defines how many)
        self.transformer_encoder = TransformerEncoder(encoder_layer, num_encoder_layers)
        # Create a single layer of Transformer decoder
        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout)
        # Stack multiple decoder layers (num_decoder_layers defines how many)
        self.transformer_decoder = TransformerDecoder(decoder_layer, num_decoder_layers)

        # Final linear layer to map the decoder output to the target vocabulary size
        # The output dimension is the size of the target vocabulary (tgt_vocab_size)
        self.fc_out = nn.Linear(d_model, tgt_vocab_size)

        # Store the model embedding dimension (d_model) for scaling later
        self.d_model = d_model

    def forward(
        self,
        src,                    # Source sequence (input)
        tgt,                    # Target sequence (input for the decoder)
        src_mask,               # Mask for the source sequence (to avoid attending to padding tokens)
        tgt_mask,               # Mask for the target sequence (prevents attention to future tokens)
        src_padding_mask,       # Padding mask for the source (to avoid attention to padding)
        tgt_padding_mask,       # Padding mask for the target (to avoid attention to padding)
        memory_key_padding_mask # Padding mask for the memory (encoder output) in the decoder
    ):
        # Embed the source sequence and scale by sqrt of d_model for stable gradients
        src_emb = self.src_embedding(src) * math.sqrt(self.d_model)

        # Embed the target sequence and scale by sqrt of d_model
        tgt_emb = self.tgt_embedding(tgt) * math.sqrt(self.d_model)

        # Add positional encodings to the source embeddings
        src_emb = self.positional_encoding(src_emb)
        # Add positional encodings to the target embeddings
        tgt_emb = self.positional_encoding(tgt_emb)
        # Pass the source embeddings through the Transformer encoder
        # The encoder produces a memory representation for the source sequence
        memory = self.transformer_encoder(src_emb, mask=src_mask, src_key_padding_mask=src_padding_mask)
        # Pass the target embeddings and the memory (encoder output) through the Transformer decoder
        # The decoder attends to both the target sequence and the encoder's memory
        output = self.transformer_decoder(tgt_emb, memory, tgt_mask=tgt_mask, source_mask = src_mask)
        # Apply the final linear layer to map the decoder output to the target vocabulary
        output = self.fc_out(output)
        # Return the final output (logits over the target vocabulary)
        return output


