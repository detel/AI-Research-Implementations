import jax
import jax.numpy as jnp
from functools import partial

def l2_distance(queries: jax.Array, database: jax.Array) -> jax.Array:
    """
    Computes pairwise squared L2 distances.
    
    Why expand (x - y)^2 to x^2 - 2xy + y^2?
    Computing the difference directly via broadcasting e.g., 
    `(queries[:, None, :] - database[None, :, :]) ** 2` creates a massive intermediate 
    tensor of shape (B, N, D). This scales terribly and rapidly causes OOM (Out of Memory) 
    errors for large datasets. By expanding the square, we leverage highly optimized matrix 
    multiplication (`jnp.dot`) for the `2xy` term, meaning we only materialize tensors up 
    to size (B, N). This drastically reduces memory bandwidth and accelerates computation.
    
    Args:
        queries: Array of shape (B, D) where B is batch size and D is dimension.
        database: Array of shape (N, D) where N is database size and D is dimension.
        
    Returns:
        Array of shape (B, N) containing squared L2 distances.
    """
    # (x - y)^2 = x^2 - 2xy + y^2
    q_sq = jnp.sum(queries ** 2, axis=-1, keepdims=True)  # (B, 1)
    d_sq = jnp.sum(database ** 2, axis=-1)  # (N,)
    dot = jnp.dot(queries, database.T)  # (B, N)
    
    dist = q_sq - 2 * dot + d_sq
    # Clip to 0 to avoid negative values due to floating point inaccuracies
    return jnp.maximum(dist, 0.0)

def cosine_distance(queries: jax.Array, database: jax.Array) -> jax.Array:
    """
    Computes pairwise cosine distances (1 - cosine similarity).
    
    Args:
        queries: Array of shape (B, D).
        database: Array of shape (N, D).
        
    Returns:
        Array of shape (B, N) containing cosine distances.
    """
    # Add small epsilon to avoid division by zero
    eps = 1e-8
    q_norm = queries / (jnp.linalg.norm(queries, axis=-1, keepdims=True) + eps)
    d_norm = database / (jnp.linalg.norm(database, axis=-1, keepdims=True) + eps)
    
    sim = jnp.dot(q_norm, d_norm.T)  # (B, N)
    
    # Cosine distance is typically 1 - similarity
    # Clip to avoid values slightly less than 0
    return jnp.maximum(1.0 - sim, 0.0)

def dot_product_distance(queries: jax.Array, database: jax.Array) -> jax.Array:
    """
    Computes pairwise dot product similarity.
    Note: For dot product, larger values mean higher similarity.
    
    Args:
        queries: Array of shape (B, D).
        database: Array of shape (N, D).
        
    Returns:
        Array of shape (B, N) containing dot product similarities.
    """
    return jnp.dot(queries, database.T)

@partial(jax.jit, static_argnames=['k', 'metric'])
def vector_search(queries: jax.Array, database: jax.Array, k: int, metric: str = 'l2'):
    """
    Finds the top-K nearest neighbors.
    
    Why use static_argnames=['k', 'metric'] and standard if/elif?
    In JAX, regular Python control flow (if/elif) fails if conditions depend on traced 
    tensors. By marking `metric` and `k` as static, we tell JAX to treat them as constant 
    Python values at compile time. This permits the use of simple `if/elif` statements to 
    dynamically construct only the specific computation graph needed (e.g., tracing only 
    the L2 code path). Using `jax.lax.cond` or `jax.lax.switch` instead would trace and 
    compile all branches into the device program, wasting compile time and device memory. 
    Additionally, `k` must be static so JAX knows the output array shapes statically.
    
    When to use which metric?
    - 'l2': Standard Euclidean spatial search. Use when exact spatial distance and magnitude matter.
    - 'cosine': Measures angular orientation. Use when vectors represent directions or 
      when magnitude should not influence similarity (e.g., standardized text embeddings).
    - 'dot_product': Raw inner product (Maximum Inner Product Search / MIPS). Use when 
      magnitude is a deliberate signal of importance/confidence (e.g., recommender systems, 
      attention mechanisms).
      
    Args:
        queries: Array of shape (B, D).
        database: Array of shape (N, D).
        k: The number of nearest neighbors to retrieve.
        metric: The distance metric to use ('l2', 'cosine', or 'dot_product').
        
    Returns:
        distances: Array of shape (B, k) containing distances (or similarities).
        indices: Array of shape (B, k) containing indices of the top-k vectors in the database.
    """
    if metric == 'l2':
        # For L2, smaller distance is better.
        # jax.lax.top_k returns largest values, so we negate distances.
        dists = l2_distance(queries, database)
        scores = -dists
    elif metric == 'cosine':
        # For Cosine distance, smaller distance (1-sim) is better.
        dists = cosine_distance(queries, database)
        scores = -dists
    elif metric == 'dot_product':
        # For Dot Product, larger similarity is typically better.
        dists = dot_product_distance(queries, database)
        scores = dists
    else:
        # Note: exceptions in jitted functions should generally be avoided if they
        # depend on dynamic values, but metric is a static_argname, so this is safe.
        raise ValueError(f"Unknown metric: {metric}")
    
    # Retrieve top-K scores and their corresponding indices
    top_k_scores, top_k_indices = jax.lax.top_k(scores, k)
    
    # Convert scores back to original distance/similarity domain
    if metric in ['l2', 'cosine']:
        top_k_dists = -top_k_scores
    else:
        top_k_dists = top_k_scores
        
    return top_k_dists, top_k_indices
