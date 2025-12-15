
import torch
import torch.nn.functional as F
from flash_attention import flash_attention_forward

def manual_attention(Q, K, V):
    d_k = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / (d_k ** 0.5)
    p_attn = F.softmax(scores, dim=-1)
    return torch.matmul(p_attn, V)

def test_flash_attention():
    torch.manual_seed(42)
    N = 128
    d = 64
    
    Q = torch.randn(N, d)
    K = torch.randn(N, d)
    V = torch.randn(N, d)
    
    print("Running Flash Attention Simulation...")
    out_flash = flash_attention_forward(Q, K, V, block_size_r=32, block_size_c=32)
    
    print("Running Standard Attention...")
    out_standard = manual_attention(Q, K, V)
    
    # Compare
    diff = torch.abs(out_flash - out_standard).max()
    print(f"Max difference: {diff.item()}")
    
    if diff < 1e-4:
        print("Test PASSED: Outputs match!")
    else:
        print("Test FAILED: Outputs do not match.")

if __name__ == "__main__":
    test_flash_attention()
