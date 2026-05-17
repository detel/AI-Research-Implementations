import jax
import jax.numpy as jnp
from functools import partial
from jax.experimental.shard_map import shard_map
from jax.sharding import Mesh, PartitionSpec as P

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

@partial(jax.jit, static_argnames=['k', 'metric'])
def distributed_vector_search(mesh: Mesh, queries: jax.Array, database: jax.Array, k: int, metric: str = 'l2'):
    """
    Finds the top-K nearest neighbors using manual distributed SPMD compute.
    
    This function explicitly maps local search operations across the given device mesh using `jax.shard_map`.
    
    Args:
        mesh: The device Mesh to shard across.
        queries: Array of shape (B, D). Must be perfectly replicated across the mesh.
        database: Array of shape (N, D). Must be sharded along its first dimension (rows).
        k: The number of nearest neighbors to retrieve.
        metric: The distance metric to use ('l2', 'cosine', or 'dot_product').
        
    Returns:
        distances: Array of shape (B, k) containing distances.
        indices: Array of shape (B, k) containing global indices of the top-k vectors.
    """
    
    def local_search(local_queries, local_db):
        # 1. Local Distances Calculation
        if metric == 'l2':
            dists = l2_distance(local_queries, local_db)
            scores = -dists
        elif metric == 'cosine':
            dists = cosine_distance(local_queries, local_db)
            scores = -dists
        elif metric == 'dot_product':
            dists = dot_product_distance(local_queries, local_db)
            scores = dists
        else:
            raise ValueError(f"Unknown metric: {metric}")
        
        # 2. Local top-K
        local_top_scores, local_top_indices = jax.lax.top_k(scores, k)
        
        # 3. Apply global index offset
        # Find which shard we are on (device index)
        device_id = jax.lax.axis_index('data')
        
        # Determine the number of local rows to offset the indices
        # local_db.shape[0] gives the size of the database chunk residing on this specific device.
        # e.g., if total DB is 8000 and we have 8 devices:
        # local_db_size = 1000
        local_db_size = local_db.shape[0]
        
        # We need to map the local indices (which range from 0 to 999) back to their 
        # original global indices (which range from 0 to 7999).
        # e.g., device 0 has indices 0-999 (offset = 0 * 1000 = 0)
        #       device 1 has indices 1000-1999 (offset = 1 * 1000 = 1000)
        global_indices = local_top_indices + device_id * local_db_size
        
        # 4. Gather local results from all devices
        # all_gather takes an array from each device and concatenates/stacks them along a new axis.
        # It's a collective communication operation where every device shares its local top-K 
        # results with every other device in the mesh.
        #
        # Example with num_devices=2, B=1 (1 query), k=2 (top 2):
        # Device 0 finds local scores [[0.9, 0.8]] and global indices [[10, 20]]
        # Device 1 finds local scores [[0.95, 0.7]] and global indices [[1015, 1025]]
        # 
        # all_gather stacks them so EVERY device gets the full tensor of shape (2, 1, 2):
        # gathered_scores = jnp.array([
        #   [[0.9, 0.8]],    # Device 0's contribution
        #   [[0.95, 0.7]]    # Device 1's contribution
        # ])
        gathered_scores = jax.lax.all_gather(local_top_scores, axis_name='data')
        gathered_indices = jax.lax.all_gather(global_indices, axis_name='data')
        
        return gathered_scores, gathered_indices

    # Run the local_search mapping over the 'data' mesh axis
    # jax.experimental.shard_map allows us to write Single-Program, Multiple-Data (SPMD) code.
    # It takes the global arrays, splits them according to in_specs, runs `local_search` on 
    # each device concurrently using only its local slice of data, and then reassembles the 
    # results according to out_specs.
    gathered_scores, gathered_indices = shard_map(
        local_search,
        mesh=mesh,
        # in_specs: Queries are copied entirely to all devices. Database is split along axis 0.
        in_specs=(P(None, None), P('data', None)),
        # out_specs: The gathered outputs from local_search are replicated across all devices.
        out_specs=(P(None, None, None), P(None, None, None))
    )(queries, database)
    
    # 5. Global Reduction (Absolute Top-K)
    # The output from shard_map is currently shaped (num_devices, B, k)
    # We want to find the top-K across all device chunks, so we reshape to (B, num_devices * k)
    B = queries.shape[0]
    
    # We swap axes so Batch is first, then reshape to flatten device and K dimensions.
    # We want to combine the top-K candidates from all devices into a single pool for each query.
    #
    # Continuing our 2-device, B=1, k=2 example:
    # gathered_scores shape: (2, 1, 2)
    # [
    #   [[0.9, 0.8]],  # Device 0
    #   [[0.95, 0.7]]  # Device 1
    # ]
    #
    # 1. transpose(1, 0, 2) brings Batch to the front -> shape (1, 2, 2):
    # [
    #   [ [0.9, 0.8], [0.95, 0.7] ]
    # ]
    #
    # 2. reshape(B, -1) flattens the devices and candidates -> shape (1, 4):
    # [[0.9, 0.8, 0.95, 0.7]]
    #
    # Now we have all 4 candidate scores (and indices) in a flat array, ready for a final top_k!
    flat_gathered_scores = gathered_scores.transpose(1, 0, 2).reshape(B, -1)
    flat_gathered_indices = gathered_indices.transpose(1, 0, 2).reshape(B, -1)
    
    # Run a final top-k over all the gathered local top-k candidates
    global_top_k_scores, global_top_k_idx = jax.lax.top_k(flat_gathered_scores, k)
    
    # Map the relative indices from the global_top_k_idx back to the actual database global indices
    final_indices = jnp.take_along_axis(flat_gathered_indices, global_top_k_idx, axis=-1)
    
    # Convert scores back to distances if necessary
    if metric in ['l2', 'cosine']:
        final_dists = -global_top_k_scores
    else:
        final_dists = global_top_k_scores
        
    return final_dists, final_indices
