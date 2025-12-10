
import torch
from scaled_dot_product_attention import ScaledDotProductAttention

def test_scaled_dot_product_attention():
    # Dimensions
    batch_size = 2
    seq_len = 10
    d_model = 64
    d_k = 32
    d_v = 32
    
    # Initialize implementation
    try:
        attention = ScaledDotProductAttention(d_model=d_model, d_k=d_k, d_v=d_v)
        print("Initialization successful.")
    except Exception as e:
        print(f"Initialization failed: {e}")
        return

    # Create dummy input
    x = torch.randn(batch_size, seq_len, d_model)
    
    # Forward pass
    try:
        output = attention(x)
        print("Forward pass successful.")
        print(f"Output shape: {output.shape}")
        
        # Expected shape: (batch_size, seq_len, d_v) because:
        # attention_weights: (B, L, L)
        # V: (B, L, d_v)
        # result: (B, L, d_v)
        
        expected_shape = (batch_size, seq_len, d_v)
        if output.shape == expected_shape:
            print("Shape check passed.")
        else:
            print(f"Shape check failed. Expected {expected_shape}, got {output.shape}")
            
    except Exception as e:
        print(f"Forward pass failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_scaled_dot_product_attention()
