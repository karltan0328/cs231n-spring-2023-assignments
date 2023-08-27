import torch
import torch.nn as nn
from torch.nn import functional as F
import math

"""
This file defines layer types that are commonly used for transformers.
该文件定义了transformers中常用的层类型。
"""

class PositionalEncoding(nn.Module):
    """
    Encodes information about the positions of the tokens in the sequence. In
    this case, the layer has no learnable parameters, since it is a simple
    function of sines and cosines.
    对有关令牌在序列中的位置的信息进行编码。在这种情况下，层没有可学习的参数，因为它是正弦和余弦的简单函数。
    """
    def __init__(self, embed_dim, dropout=0.1, max_len=5000):
        """
        Construct the PositionalEncoding layer.
        构造PositionalEncoding层。

        Inputs:
        - embed_dim: the size of the embed dimension
          嵌入维度的大小
        - dropout: the dropout value
          丢弃值
        - max_len: the maximum possible length of the incoming sequence
          将来序列的最大可能长度
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        assert embed_dim % 2 == 0
        # Create an array with a "batch dimension" of 1 (which will broadcast
        # across all examples in the batch).
        pe = torch.zeros(1, max_len, embed_dim)
        ############################################################################
        # TODO: Construct the positional encoding array as described in            #
        # Transformer_Captioning.ipynb.  The goal is for each row to alternate     #
        # sine and cosine, and have exponents of 0, 0, 2, 2, 4, 4, etc. up to      #
        # embed_dim. Of course this exact specification is somewhat arbitrary, but #
        # this is what the autograder is expecting. For reference, our solution is #
        # less than 5 lines of code.                                               #
        # 构造如Transformer_Captioning.ipynb中所述的位置编码数组。
        # 目标是每行交替正弦和余弦，并且指数为0,0,2,2,4,4等，直到embed_dim。
        # 当然，这个确切的规范有点武断，但这是自动分级器所期望的。参考我们的解决方案少于5行代码。
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        for j in range(embed_dim):
            for i in range(max_len):
                if j % 2 == 0:
                    pe[0, i, j] = math.sin(i * 10000 ** (-j / embed_dim))
                else:
                    pe[0, i, j] = math.cos(i * 10000 ** (-(j - 1) / embed_dim))
        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # Make sure the positional encodings will be saved with the model
        # parameters (mostly for completeness).
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Element-wise add positional embeddings to the input sequence.
        对输入序列进行逐元素加位置嵌入。

        Inputs:
        - x: the sequence fed to the positional encoder model, of shape
            (N, S, D), where N is the batch size, S is the sequence length and
            D is embed dim
          馈送到位置编码器模型的序列，形状为(N, S, D)，其中N是批量大小，S是序列长度，D是嵌入维度

        Returns:
        - output: the input sequence + positional encodings, of shape (N, S, D)
          输入序列+位置编码，形状为(N, S, D)
        """
        N, S, D = x.shape
        # Create a placeholder, to be overwritten by your code below.
        output = torch.empty((N, S, D))
        ############################################################################
        # TODO: Index into your array of positional encodings, and add the         #
        # appropriate ones to the input sequence. Don't forget to apply dropout    #
        # afterward. This should only take a few lines of code.                    #
        # 索引到位置编码的数组中，并将适当的位置编码添加到输入序列中。不要忘记之后应用丢弃。
        # 这应该只需要几行代码。
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        output = x + self.pe[:, :S, :]
        output = self.dropout(output)
        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################
        return output


class MultiHeadAttention(nn.Module):
    """
    A model layer which implements a simplified version of masked attention, as
    introduced by "Attention Is All You Need" (https://arxiv.org/abs/1706.03762).
    一个模型层，它实现了掩码注意力的简化版本，如“Attention Is All You Need”(https://arxiv.org/abs/1706.03762)所介绍的那样。

    Usage:
      attn = MultiHeadAttention(embed_dim, num_heads=2)

      # self-attention
      data = torch.randn(batch_size, sequence_length, embed_dim)
      self_attn_output = attn(query=data, key=data, value=data)

      # attention using two inputs
      other_data = torch.randn(batch_size, sequence_length, embed_dim)
      attn_output = attn(query=data, key=other_data, value=other_data)
    """

    def __init__(self, embed_dim, num_heads, dropout=0.1):
        """
        Construct a new MultiHeadAttention layer.
        构造一个新的MultiHeadAttention层。

        Inputs:
        - embed_dim: Dimension of the token embedding
          令牌嵌入的维度
        - num_heads: Number of attention heads
          注意力头的数量
        - dropout: Dropout probability
          丢弃概率
        """
        super().__init__()
        assert embed_dim % num_heads == 0

        # We will initialize these layers for you, since swapping the ordering
        # would affect the random number generation (and therefore your exact
        # outputs relative to the autograder). Note that the layers use a bias
        # term, but this isn't strictly necessary (and varies by
        # implementation).
        self.key = nn.Linear(embed_dim, embed_dim)
        self.query = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.proj = nn.Linear(embed_dim, embed_dim)

        self.attn_drop = nn.Dropout(dropout)

        self.n_head = num_heads
        self.emd_dim = embed_dim
        self.head_dim = self.emd_dim // self.n_head

    def forward(self, query, key, value, attn_mask=None):
        """
        Calculate the masked attention output for the provided data, computing
        all attention heads in parallel.
        计算所提供数据的掩码注意力输出，同时计算所有注意力头。

        In the shape definitions below, N is the batch size, S is the source
        sequence length, T is the target sequence length, and E is the embedding
        dimension.
        在下面的形状定义中，N是批量大小，S是源序列长度，T是目标序列长度，E是嵌入维度。

        Inputs:
        - query: Input data to be used as the query, of shape (N, S, E)
          用作查询的输入数据，形状为(N, S, E)
        - key: Input data to be used as the key, of shape (N, T, E)
          用作键的输入数据，形状为(N, T, E)
        - value: Input data to be used as the value, of shape (N, T, E)
          用作值的输入数据，形状为(N, T, E)
        - attn_mask: Array of shape (S, T) where mask[i,j] == 0 indicates token
          i in the source should not influence token j in the target.
          形状为(S, T)的数组，其中mask[i,j]==0表示源中的令牌i不应影响目标中的令牌j。

        Returns:
        - output: Tensor of shape (N, S, E) giving the weighted combination of
          data in value according to the attention weights calculated using key
          and query.
          给出根据使用键和查询计算的注意力权重对值中的数据进行加权组合的形状为(N, S, E)的张量。
        """
        N, S, E = query.shape
        N, T, E = value.shape
        # Create a placeholder, to be overwritten by your code below.
        output = torch.empty((N, S, E))
        ############################################################################
        # TODO: Implement multiheaded attention using the equations given in       #
        # Transformer_Captioning.ipynb.                                            #
        # A few hints:                                                             #
        #  1) You'll want to split your shape from (N, T, E) into (N, T, H, E/H),  #
        #     where H is the number of heads.                                      #
        #  2) The function torch.matmul allows you to do a batched matrix multiply.#
        #     For example, you can do (N, H, T, E/H) by (N, H, E/H, T) to yield a  #
        #     shape (N, H, T, T). For more examples, see                           #
        #     https://pytorch.org/docs/stable/generated/torch.matmul.html          #
        #  3) For applying attn_mask, think how the scores should be modified to   #
        #     prevent a value from influencing output. Specifically, the PyTorch   #
        #     function masked_fill may come in handy.                              #
        # 实现使用Transformer_Captioning.ipynb中给出的方程的多头注意力。
        # 一些提示:
        # 1)你会想把你的形状从(N, T, E)分成(N, T, H, E/H)，其中H是头的数量。
        # 2)函数torch.matmul允许您执行批量矩阵乘法。
        #   例如，您可以通过(N, H, T, E/H)乘以(N, H, E/H, T)来产生一个形状(N, H, T, T)。
        #   有关更多示例，请参见https://pytorch.org/docs/stable/generated/torch.matmul.html
        # 3)对于应用attn_mask，想想如何修改分数以防止一个值影响输出。
        # 具体来说，PyTorch函数masked_fill可能会派上用场。
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        H = self.n_head
        key_v = self.key(key).reshape(N, T, H, self.head_dim).transpose(1, 2)       # (N, T, H, E/H) -> (N, H, T, E/H)
        query_v = self.query(query).reshape(N, S, H, self.head_dim).transpose(1, 2) # (N, S, H, E/H) -> (N, H, S, E/H)
        value_v = self.value(value).reshape(N, T, H, self.head_dim).transpose(1, 2) # (N, T, H, E/H) -> (N, H, T, E/H)

        e = torch.matmul(query_v, key_v.transpose(2, 3)) / math.sqrt(self.head_dim) # (N, H, S, E/H) * (N, H, E/H, T) -> (N, H, S, T)
        if attn_mask != None:
            e = torch.masked_fill(e, attn_mask == 0, -math.inf)
        a = nn.Softmax(dim=-1)(e)
        a = self.attn_drop(a)
        y = torch.matmul(a, value_v) # (N, H, S, T) * (N, H, T, E/H) -> (N, H, S, E/H)
        output = self.proj(y.transpose(1, 2).reshape(N, S, self.emd_dim))
        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################
        return output
