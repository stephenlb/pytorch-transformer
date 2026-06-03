class PositionalEncoding(nn.Module):
    """
    Adds sinusoidal positional encoding to embeddings.

    This layer encodes the absolute position of tokens in a sequence using sine
    and cosine functions of different frequencies. The positional encoding is added
    to the embeddings to give the model information about token positions, since
    the self-attention mechanism alone does not capture sequential order.

    Args:
        d_model (int): The dimension of the embedding vectors
        seq_len (int): The maximum sequence length to encode
        dropout (float): Dropout rate applied to the output
    """

    def __init__(self, d_model: int, seq_len: int, dropout: float):
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        # provide dropout
        self.dropout = nn.Dropout(dropout)

        # create matrix of shape (seq_len, d_model)
        p_e = torch.zeros(seq_len, d_model)

        # create a vector of shape (seq_len)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(
            1
        )  # (seq_len, 1)

        # create a vector of shape (d_model)
        # div_term = 1 / 10000^(2i/d_model)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        # apply sine to even indices
        # PE(pos, 2i) = sin(pos/10000^(2i/d_model))
        p_e[:, 0::2] = torch.sin(position * div_term)

        # apply cos to odd indices
        # PE(pos,2i+1) = cos(pos/10000^(2i/d_model))
        p_e[:, 1::2] = torch.cos(position * div_term)

        # add extra dim for batch in p_e
        p_e = p_e.unsqueeze(0)  # (1, seq_len, d_model)

        # register positional encoding as a buffer
        self.register_buffer("p_e", p_e)

    def forward(self, x):
        # positional encoding = embedding + sinusoidal positional encoding
        x = x + (
            self.p_e[:, : x.shape[1], :].requires_grad_(False)
        )  # (batch, seq_len, d_model)
        return self.dropout(x)
