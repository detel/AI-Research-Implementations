# --- Softmax ---
# Formula: S_i = e^(z_i) / sum(e^(z_j))
def run_softmax(in_features, dim):
    """Apply softmax along specified dimension"""
    return torch.softmax(in_features, dim=dim)

# --- Sigmoid Linear Unit (SiLU) / Swish ---
# Formula: SiLU(x) = x * sigmoid(x)
def run_silu(in_features):
    """SiLU activation: x * sigmoid(x)"""
    return in_features * torch.sigmoid(in_features)

