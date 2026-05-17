import jax
import jax.numpy as jnp

def init_resnet_block_params(key, in_channels, out_channels, stride=1):
    """
    Initializes the parameters for a ResNet block.
    
    In raw JAX, we keep parameters completely separate from the computation.
    Furthermore, JAX requires strict separation between:
      1. `params`: Differentiable weights updated by the optimizer via gradients.
      2. `state`: Non-differentiable mutable state updated via forward pass 
                  logic (like batch norm moving averages).
    
    Args:
        key: jax.random.PRNGKey for random initialization.
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        stride: Stride for the first convolution.
        
    Returns:
        params: Dictionary of differentiable weights/biases.
        state: Dictionary of non-differentiable tracking metrics (BN stats).
    """
    # TODO: Implement parameter initialization. You will need:
    # 1. Conv layer 1 weights (3x3 kernel, in_channels, out_channels)
    # 2. BatchNorm layer 1 parameters (gamma, beta) and running stats (mean, variance)
    # 3. Conv layer 2 weights (3x3 kernel, out_channels, out_channels)
    # 4. BatchNorm layer 2 parameters and running stats
    # 5. Optional: Projection layer (Conv 1x1) if in_channels != out_channels or stride != 1
    # 
    # Hint: Use jax.random.normal or jax.random.uniform for weights.
    
    # Split the key into 4 different keys for each parameter
    key, k1, k2, k3 = jax.random.split(key, 4)

    # He initialization for weights 
    fan_in_1 = 3 * 3 * in_channels
    std1 = jnp.sqrt(2. / fan_in_1)

    # He initialization for weights 
    fan_in_2 = 3 * 3 * out_channels
    std2 = jnp.sqrt(2. / fan_in_2)

    # ── 1. Differentiable Parameters (params) ────────────────────────
    # These are the weights and biases that will receive gradients from jax.grad 
    # and be updated by an optimizer (like Adam/SGD).
    params = {
        'conv_layer_1':{
            'weight': std1 * jax.random.normal(k1, (3, 3, in_channels, out_channels)),
            'bias': jnp.zeros(out_channels),
        },
        'batch_norm_1':{
            'gamma': jnp.ones(out_channels),
            'beta': jnp.zeros(out_channels),
        },
        'conv_layer_2':{
            'weight': std2 * jax.random.normal(k2, (3, 3, out_channels, out_channels)),
            'bias': jnp.zeros(out_channels),
        }, 
        'batch_norm_2':{
            'gamma': jnp.ones(out_channels),
            'beta': jnp.zeros(out_channels),
        },
        'stride': stride
    }

    # ── 2. Non-differentiable State (state) ──────────────────────────
    # These are moving averages tracked over time. They DO NOT receive gradients.
    # If we put them in `params`, jax.grad would try to differentiate them, 
    # causing mathematical errors and wasting memory.
    state = {
        'batch_norm_1':{
            'running_mean': jnp.zeros(out_channels),
            'running_var': jnp.ones(out_channels)
        },
        'batch_norm_2':{
            'running_mean': jnp.zeros(out_channels),
            'running_var': jnp.ones(out_channels)
        }
    }

    if in_channels != out_channels or stride != 1:
        fan_in_proj = 1 * 1 * in_channels
        std_proj = jnp.sqrt(2. / fan_in_proj)
        params['projection'] = {
            'weight': std_proj * jax.random.normal(k3, (1, 1, in_channels, out_channels)),
            'bias': jnp.zeros(out_channels),
        }
    else:
        params['projection'] = None
        
    return params, state

def relu(x):
    """
    Applies the ReLU activation function.
    """
    return jnp.maximum(x, 0)

def conv2d(x, weight, bias=None, stride=1, padding='SAME'):
    """
    Applies a 2D convolution using an im2col approach with jnp operations.

    Instead of calling jax.lax.conv_general_dilated we:
      1. Pad the input with jnp.pad (for SAME padding).
      2. Extract patches for each kernel position via strided jnp array slicing.
      3. Concatenate patches along the channel axis to form the im2col matrix.
      4. Dot-product with the flattened kernel using jnp.dot.

    Args:
        x: Input tensor of shape (batch_size, height, width, in_channels).
        weight: Convolution weights of shape (kH, kW, in_channels, out_channels).
        bias: Optional bias tensor of shape (out_channels,).
        stride: Integer, stride of the convolution.
        padding: 'SAME' or 'VALID'.

    Returns:
        Output tensor of shape (batch_size, out_height, out_width, out_channels).
    """
    batch_size, h_in, w_in, c_in = x.shape
    kh, kw, _, c_out = weight.shape

    # --- Compute output dimensions and pad if needed ---
    if padding == 'SAME':
        out_h = (h_in + stride - 1) // stride          # ceil(h_in / stride)
        out_w = (w_in + stride - 1) // stride
        pad_h = max((out_h - 1) * stride + kh - h_in, 0)
        pad_w = max((out_w - 1) * stride + kw - w_in, 0)
        pad_top, pad_bottom = pad_h // 2, pad_h - pad_h // 2
        pad_left, pad_right = pad_w // 2, pad_w - pad_w // 2
        x = jnp.pad(x, ((0, 0), (pad_top, pad_bottom), (pad_left, pad_right), (0, 0)))
    elif padding == 'VALID':
        out_h = (h_in - kh) // stride + 1
        out_w = (w_in - kw) // stride + 1
    else:
        raise ValueError(f"Unsupported padding mode: {padding}")

    # --- im2col via strided slicing ---
    # For each (i, j) in the kernel we grab every output-position at the
    # correct stride, giving a (batch, out_h, out_w, c_in) slice.
    patches = []
    for i in range(kh):
        for j in range(kw):
            patch = x[:, i:i + out_h * stride:stride,
                         j:j + out_w * stride:stride, :]
            patches.append(patch)

    # Stack along the channel axis: (batch, out_h, out_w, kh * kw * c_in)
    patches = jnp.concatenate(patches, axis=-1)

    # --- Dot product with reshaped kernel ---
    # weight: (kh, kw, c_in, c_out) -> (kh * kw * c_in, c_out)
    w_col = weight.reshape(-1, c_out)
    out = jnp.dot(patches, w_col)          # (batch, out_h, out_w, c_out)

    if bias is not None:
        out = out + bias

    return out

def batch_norm(x, gamma, beta, running_mean, running_var, is_training=True, eps=1e-5, momentum=0.9):
    """
    Applies Batch Normalization.
    
    Args:
        x: Input tensor.
        gamma: Scale parameter.
        beta: Shift parameter.
        running_mean: Running mean of the features.
        running_var: Running variance of the features.
        is_training: Boolean, whether in training mode (updates stats) or inference mode.
        eps: Small constant for numerical stability.
        momentum: Momentum for updating running stats.
        
    Returns:
        out: Normalized tensor.
        new_running_mean: Updated running mean.
        new_running_var: Updated running var.
    """
    # TODO: Implement Batch Normalization using raw jax.numpy operations.
    if is_training:
        # Calculate mean and variance along batch and spatial dimensions (N, H, W)
        mean = jnp.mean(x, axis=(0, 1, 2))
        var = jnp.var(x, axis=(0, 1, 2))
        
        # Update running statistics
        new_running_mean = momentum * running_mean + (1.0 - momentum) * mean
        new_running_var = momentum * running_var + (1.0 - momentum) * var
    else:
        mean = running_mean
        var = running_var
        new_running_mean = running_mean
        new_running_var = running_var
        
    # Normalize the input `x` using the computed stats.
    x_norm = (x - mean) / jnp.sqrt(var + eps)
    
    # Scale and shift the normalized input using `gamma` and `beta`.
    out = gamma * x_norm + beta
    
    return out, new_running_mean, new_running_var

def resnet_block_forward(params, state, x, is_training=True):
    """
    Forward pass of the ResNet Block.
    
    Args:
        params: Dictionary of differentiable parameters.
        state: Dictionary of mutable state (BN running stats).
               We pass this explicitly because JAX forbids in-place mutation.
        x: Input tensor of shape (batch_size, height, width, in_channels).
        is_training: Boolean, whether in training mode.
        
    Returns:
        out: Output tensor after the ResNet block.
        new_state: Updated dictionary of mutable state. 
                   Must be returned so the training loop doesn't lose the updates.
    """
    stride = params.get('stride', 1)
    identity = x

    # --- Main branch ---
    # Conv1 (stride applied here) -> BN1 -> ReLU
    out = conv2d(x,
                 params['conv_layer_1']['weight'],
                 params['conv_layer_1']['bias'],
                 stride=stride)
    out, new_rm1, new_rv1 = batch_norm(
        out,
        params['batch_norm_1']['gamma'],
        params['batch_norm_1']['beta'],
        state['batch_norm_1']['running_mean'],
        state['batch_norm_1']['running_var'],
        is_training=is_training
    )
    out = relu(out)

    # Conv2 (stride=1) -> BN2
    out = conv2d(out,
                 params['conv_layer_2']['weight'],
                 params['conv_layer_2']['bias'],
                 stride=1)
    out, new_rm2, new_rv2 = batch_norm(
        out,
        params['batch_norm_2']['gamma'],
        params['batch_norm_2']['beta'],
        state['batch_norm_2']['running_mean'],
        state['batch_norm_2']['running_var'],
        is_training=is_training
    )

    # --- Skip / shortcut connection ---
    if params['projection'] is not None:
        identity = conv2d(identity,
                          params['projection']['weight'],
                          params['projection']['bias'],
                          stride=stride)

    # Residual add + final activation
    out = relu(out + identity)

    # --- Create updated state ---
    new_state = {
        'batch_norm_1': {
            'running_mean': new_rm1,
            'running_var': new_rv1,
        },
        'batch_norm_2': {
            'running_mean': new_rm2,
            'running_var': new_rv2,
        }
    }

    return out, new_state

if __name__ == "__main__":
    # ── Master PRNG key ──────────────────────────────────────────────
    # Every random operation receives its own unique subkey via
    # jax.random.split so that no key is ever reused.
    master_key = jax.random.PRNGKey(0)

    batch_size = 4
    in_channels = 16
    out_channels = 32
    height, width = 32, 32
    num_iterations = 3  # number of simulated training iterations

    # ── Split: one key for parameter init, one for the iteration loop ─
    init_key, loop_key = jax.random.split(master_key)

    print("Initializing parameters and state...")
    params, state = init_resnet_block_params(init_key, in_channels, out_channels, stride=2)

    # ── Explicit iteration loop with proper key threading ────────────
    for step in range(num_iterations):
        # Split the loop key *before* each iteration:
        #   • loop_key is advanced for the next iteration
        #   • step_key is consumed for this iteration's random data
        loop_key, step_key = jax.random.split(loop_key)

        # Generate a fresh random batch for this iteration
        x = jax.random.normal(step_key, (batch_size, height, width, in_channels))

        print(f"\n── Iteration {step} ──")
        out, state = resnet_block_forward(params, state, x, is_training=True)
        print(f"  Input shape:  {x.shape}")
        print(f"  Output shape: {out.shape}")

    print("\nAll iterations completed successfully!")
