
from .linear import Linear
from .scaled_dot_product_attention import ScaledDotProductAttention
import numpy as np
from src.nn import scaled_dot_product_attention
from src.nn import linear

class MultiHeadAttention:
    """
    Multi Head Attention
    """
    def __init__(self, embed_dim, num_heads):
        """
        :param embed_dim: Embedding dimension
        :param num_heads: Number of attention heads
        """
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")

        self.embed_dim = embed_dim
        self.num_heads = num_heads

        self.attention = scaled_dot_product_attention.ScaledDotProductAttention()

        self.q_proj   = linear.Linear(embed_dim, embed_dim)
        self.k_proj   = linear.Linear(embed_dim, embed_dim)
        self.v_proj   = linear.Linear(embed_dim, embed_dim)
        self.out_proj = linear.Linear(embed_dim, embed_dim)

    def init_weights(self, Wq, bq, Wk, bk, Wv, bv, Wo, bo):
        """
        Initialize the weights and biases with the given values.
        """
        self.q_proj.init_weights(Wq, bq)
        self.k_proj.init_weights(Wk, bk)
        self.v_proj.init_weights(Wv, bv)
        self.out_proj.init_weights(Wo, bo)

    def forward(self, query, key, value, key_padding_mask=None, attn_mask=None):
        """
        :param query: (N, L, E)
        :param key: (N, S, E)
        :param value: (N, S, E)
        :param key_padding_mask: (N, S) where 1/True indicates positions to ignore
        :param attn_mask: (L, S) where 1/True indicates positions to ignore
        :return: (N, L, E)
        """
        self.N = query.shape[0]
        self.L = query.shape[1]
        self.S = key.shape[1]
        self.E = query.shape[2]

        q = self.q_proj.forward(query)
        k = self.k_proj.forward(key)
        v = self.v_proj.forward(value)

        q = self._split_heads(q)
        k = self._split_heads(k)
        v = self._split_heads(v)

        mask = self._merge_masks(key_padding_mask, attn_mask)

        attn_outputs = self.attention.forward(q, k, v, mask)

        attn_output = self._concat_heads(attn_outputs)
        output = self.out_proj.forward(attn_output)

        return output

    def backward(self, d_output):
        """
        :param d_output: Gradient of loss wrt output of shape (N, L, E)
        :return: Gradient of loss wrt input query, key, value of shapes (N, L, E), (N, S, E), (N, S, E)
        """
        d_attn_output = self.out_proj.backward(d_output)

        d_attn_outputs = self._split_heads(d_attn_output)

        d_q, d_k, d_v = self.attention.backward(d_attn_outputs)

        d_q = self._concat_heads(d_q)
        d_k = self._concat_heads(d_k)
        d_v = self._concat_heads(d_v)

        d_q = self.q_proj.backward(d_q)
        d_k = self.k_proj.backward(d_k)
        d_v = self.v_proj.backward(d_v)
        return  d_q, d_k, d_v


    def _merge_masks(self, key_padding_mask, attn_mask):
        """
        Merge key_padding_mask and attn_mask into a single mask.
        :param key_padding_mask: (N, S)
        :param attn_mask: (L, S)
        :return: (N, H, L, S)
        """
        key_mask = key_padding_mask[:, None, None, : ]

        key_mask = np.broadcast_to(key_mask, (self.N, self.num_heads, self.L, self.S))

        attention_mask = attn_mask[None, None, :, : ]
        attention_mask = np.broadcast_to(attention_mask, (self.N, self.num_heads, self.L, self.S))

        combined_mask = key_mask | attention_mask

        return combined_mask

    def _split_heads(self, x):
        """
        Split the last dimension into (num_heads, d_k).
        Transpose to move num_heads dimension to the front.
        :param x: (N, L, embed_dim)
        :return: (N, num_heads, L, embed_dim // num_heads)
        """
        _, L_or_S, _ = x.shape
        x = x.reshape(self.N, L_or_S, self.num_heads, self.embed_dim//self.num_heads)
        x = np.swapaxes(x, 1, 2)

        return x

    def _concat_heads(self, x):
        """
        Concatenate the last dimension into (num_heads, d_k).
        Transpose to move num_heads dimension to the back.
        :param x: (N, num_heads, L, embed_dim // num_heads)
        :return: (N, L, embed_dim)
        """
        x = np.swapaxes(x, 1, 2)

        x = np.reshape(x, (self.N, self.L, self.embed_dim))

        return x       
