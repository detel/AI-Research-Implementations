
import torch
import torch.nn as nn
import math

'''
Based on Scaled Dot-Product Attention in [Attention Is All You Need (Vaswani, A., et al. (2017))](https://arxiv.org/pdf/1706.03762)
'''

class ScaledDotProductAttention(nn.Module):
    '''
    Create trainable weights for K, Q, V
    '''
    def __init__(self, d_model, d_k, d_v):
        super().__init__()
        self.d_k = d_k
        self.W_k = nn.Linear(d_model, d_k)
        self.W_q = nn.Linear(d_model, d_k)
        self.W_v = nn.Linear(d_model, d_v)
    
    '''
    x - (L, d_model)
    '''
    def forward(self, x):
        K = self.W_k(x)
        Q = self.W_q(x)
        
        # TODO: if causal attention then W_k(i, j) = float('-inf') for all i > j, i.e. don't attend to future token
        # TODO: add attention mask
        # TODO: gqa
        # TODO: Include heads
        
        # Note: In the original code, the transpose was (-2, -1) which works for batched input (B, L, H) or (L, H)
        # We perform the dot product QK^T
        # It will be (L, L) dimension.
        attention_weights = (Q @ K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # TODO: implement softmax
        # attention_weights = softmax(attention_weights)
        
        # TODO: perform last step project. Final output embedding shape should be same as input embedding shape. Also depends on number of heads.
        V = self.W_v(x)
        return attention_weights @ V