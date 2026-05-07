# AI-Research-Implementations

From-scratch implementations of foundational AI/ML papers using PyTorch and JAX — no high-level wrappers, just raw operations.

## Repository Structure

```
.
├── projects/                     # Paper-specific implementations
│   ├── attention_is_all_you_need/
│   │   └── pytorch/
│   ├── custom_resnet/
│   │   └── jax/
│   └── flash_attention/
│       └── pytorch/
└── src/                          # Shared reusable building blocks
    └── layers/
        └── pytorch/
```

---

## Projects

### 1. Attention Is All You Need — Scaled Dot-Product Attention *(PyTorch)*

> Based on [Attention Is All You Need (Vaswani et al., 2017)](https://arxiv.org/pdf/1706.03762)

| File | Description |
|------|-------------|
| [`scaled_dot_product_attention.py`](projects/attention_is_all_you_need/pytorch/scaled_dot_product_attention.py) | Implements `ScaledDotProductAttention` as an `nn.Module`. Creates trainable linear projections for Q, K, V and computes the attention output `softmax(QKᵀ / √dₖ) · V`. **WIP** — TODOs remain for causal masking, attention mask, GQA, multi-head support, custom softmax, and a final output projection. |
| [`verify_attention.py`](projects/attention_is_all_you_need/pytorch/verify_attention.py) | Smoke-test script. Creates a random `(batch=2, seq=10, d_model=64)` input, runs a forward pass, and validates the output shape is `(2, 10, d_v)`. |

---

### 2. Custom ResNet Block *(JAX)*

A full ResNet basic-block built entirely with `jax.numpy` — no `jax.lax.conv_general_dilated`, no Flax/Haiku.

| File | Description |
|------|-------------|
| [`resnet_block.py`](projects/custom_resnet/jax/resnet_block.py) | Contains all components of a ResNet basic block: |
| | • `init_resnet_block_params` — He-initialised parameter dicts for two 3×3 conv layers, two batch-norm layers, and an optional 1×1 projection shortcut. |
| | • `relu` — Element-wise ReLU via `jnp.maximum`. |
| | • `conv2d` — Manual 2D convolution using an im2col approach (strided slicing + `jnp.dot`), supporting SAME/VALID padding and arbitrary stride. |
| | • `batch_norm` — Batch normalisation with running statistics tracking (training & inference modes). |
| | • `resnet_block_forward` — Full forward pass: Conv1→BN1→ReLU→Conv2→BN2→(+shortcut)→ReLU, returning updated running stats. |
| | Includes a `__main__` test that runs a `(4, 32, 32, 16) → (4, 16, 16, 32)` forward pass. |

---

### 3. Flash Attention *(PyTorch)*

> Based on [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness (Dao et al., 2022)](https://arxiv.org/pdf/2205.14135)

| File | Description |
|------|-------------|
| [`flash_attention.py`](projects/flash_attention/pytorch/flash_attention.py) | Pure-PyTorch simulation of the FlashAttention forward pass. Implements the tiled, numerically-stable softmax algorithm with explicit `read_blocks` / `write_to_hbm` helpers to mirror the paper's HBM↔SRAM data movement. Operates on `(N, d)` tensors with configurable row/column block sizes. |
| [`test_flash_attention.py`](projects/flash_attention/pytorch/test_flash_attention.py) | Correctness test. Compares the tiled flash-attention output against a standard `softmax(QKᵀ/√d) · V` reference and asserts the max absolute difference is < 1e-4. |

---

## Shared Layers

Reusable primitives intended for use across projects.

| File | Description |
|------|-------------|
| [`activations.py`](src/layers/pytorch/activations.py) | `run_softmax(x, dim)` — softmax via `torch.softmax`. `run_silu(x)` — SiLU/Swish activation (`x · σ(x)`). |

---

## Getting Started

```bash
# Clone
git clone https://github.com/detel/AI-Research-Implementations.git
cd AI-Research-Implementations

# Run any project directly
python projects/flash_attention/pytorch/test_flash_attention.py
python projects/attention_is_all_you_need/pytorch/verify_attention.py
python projects/custom_resnet/jax/resnet_block.py
```

### Dependencies

| Framework | Used by |
|-----------|---------|
| **PyTorch** | `attention_is_all_you_need`, `flash_attention`, `src/layers` |
| **JAX** | `custom_resnet` |