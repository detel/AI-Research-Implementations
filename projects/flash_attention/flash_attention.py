"""
Based on Flash Attention in [FlashAttention: Fast and Memory-Efficient Exact Attention
with IO-Awareness (Dao, T., et al. (2022))](https://arxiv.org/pdf/2205.14135)

This is simulation in PyTorch. PyTorch is unable to work in SRAM.
TODO: Implement in CUDA
"""

import math
import torch

def read_blocks(A, start, end):
    """
    Helper to slice tensors, simulating reading a block from HBM.
    """
    return A[start:end]

def write_to_hbm(O, l, m, O_block, l_block, m_block, row_start, row_end):
    """
    Simulation of writing blocks from SRAM back to HBM.
    In actual hardware, this moves results from shared memory to global memory.
    """
    O[row_start:row_end] = O_block
    l[row_start:row_end] = l_block
    m[row_start:row_end] = m_block

def flash_attention_forward(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, block_size_r: int = 64, block_size_c: int = 64) -> torch.Tensor:
    """
    A pure PyTorch simulation of FlashAttention (Forward Pass).
    In the paper, block_size_r = ceiling(M / 4d), block_size_r = min(ceiling(M / 4d), d), where M is size of n-chip SRAM

    Args:
        Q: Query tensor of shape (N, d_k)
        K: Key tensor of shape (N, d_k)
        V: Value tensor of shape (N, d_v)
        block_size_r: Block size for rows (Query)
        block_size_c: Block size for columns (Key/Value)
    """
    N, d_k = Q.shape
    scale = 1.0 / math.sqrt(d_k)
    
    # Initialisation in High Bandwidth Memory (HBM)
    # Output
    O = torch.zeros_like(Q)
    # Running sum of exponentials. Required for softmax.
    # Initialize m to -inf
    m = torch.full((N, 1), float('-inf'), device=Q.device)
    # Initialize l to 0
    l = torch.zeros((N, 1), device=Q.device)
    # End of initialisation in HBM

    T_r = math.ceil(N / block_size_r)
    T_c = math.ceil(N / block_size_c)

    for t_j in range(T_c):
        column_start = t_j * block_size_c
        column_end = min((t_j + 1) * block_size_c, N)

        # Simulating loading K_j, V_j from HBM
        K_j = read_blocks(K, column_start, column_end)
        V_j = read_blocks(V, column_start, column_end)

        for t_i in range(T_r):
            row_start = t_i * block_size_r
            row_end = min((t_i + 1) * block_size_r, N)

            # Simulating loading Q_i, O_i, m_i, l_i from HBM
            Q_i = read_blocks(Q, row_start, row_end)
            O_i = read_blocks(O, row_start, row_end)
            m_i = read_blocks(m, row_start, row_end)
            l_i = read_blocks(l, row_start, row_end)

            # Scores. Same as scaled_dot_product_attention. 
            S_ij = scale * (Q_i @ K_j.transpose(-2, -1))

            # Find m, and l for current block
            m_ij_tilde = torch.max(S_ij, dim=-1, keepdim=True).values
            # P_ij_tilde = exp(S_ij - m_ij_tilde)  <-- unnormalized probabilities scaled by block max
            P_ij_tilde = torch.exp(S_ij - m_ij_tilde)
            l_ij_tilde = torch.sum(P_ij_tilde, dim=-1, keepdim=True)
            
            # Infer running m, and l based on previous values, and current block values. Referred as ij_tilde in the paper.
            m_i_new = torch.maximum(m_i, m_ij_tilde)
            l_i_new = (torch.exp(m_i - m_i_new) * l_i + torch.exp(m_ij_tilde - m_i_new) * l_ij_tilde)

            # Let's stick to the paper's update rule which matches the code structure intended:
            # O_i is currently the running average (normalized).
            # To update it correctly:
            # O_i_new = (l_i * e^(m_i - m_i_new) * O_i + e^(m_ij_tilde - m_i_new) * (P_ij_block @ V_j)) / l_i_new
            term1 = l_i * torch.exp(m_i - m_i_new) * O_i
            term2 = torch.exp(m_ij_tilde - m_i_new) * (P_ij_tilde @ V_j)
            O_i_new = (term1 + term2) / l_i_new

            # Write back to HBM
            write_to_hbm(O, l, m, O_i_new, l_i_new, m_i_new, row_start, row_end)
            
    return O
