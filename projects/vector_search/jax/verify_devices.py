import os
# Set up 8 simulated CPU devices before importing JAX
os.environ["XLA_FLAGS"] = '--xla_force_host_platform_device_count=8'

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec

# ==============================================================================
# JAX Distributed API Concepts
# ==============================================================================
# Mesh: 
#   A logical layout of available physical devices (like a grid of CPUs/GPUs). 
#   You map axes of this grid to human-readable names. Here, we create a 
#   1D grid with a single axis called 'data' that groups all 8 devices.
# 
# PartitionSpec: 
#   Describes how the dimensions of a multi-dimensional array map to the axes 
#   defined in your Mesh. For a 2D array, PartitionSpec('data', None) means:
#   - The first dimension (rows) maps to the 'data' axis of the Mesh (it gets sharded).
#   - The second dimension (columns) maps to `None` (it gets replicated/copied on all devices).
#
# NamedSharding: 
#   Binds a specific `Mesh` to a specific `PartitionSpec`. It creates an object 
#   that JAX can use to physically distribute data across devices when you use 
#   jax.device_put(), or to correctly shard intermediate results inside a jitted function.
# ==============================================================================

def main():
    # Verify we have 8 devices
    devices = jax.devices()
    
    print(f"Number of devices: {len(devices)}")
    # Expected output:
    # Number of devices: 8

    for d in devices:
        print(d)
        # Expected output (may vary slightly by platform):
        # TFRT_CPU_0
        # TFRT_CPU_1
        # TFRT_CPU_2
        # ...
        # TFRT_CPU_7
        
    # Define a 1D device mesh
    mesh = Mesh(jax.devices(), axis_names=('data',))
    # Visualization of the 1D Mesh:
    # A single logical axis named 'data' containing 8 devices.
    # [ CPU 0 | CPU 1 | CPU 2 | CPU 3 | CPU 4 | CPU 5 | CPU 6 | CPU 7 ]
    
    # Generate mock vector database
    num_vectors = 8000
    vector_dim = 128
    key = jax.random.PRNGKey(0)
    db_key, query_key = jax.random.split(key)
    
    mock_db = jax.random.normal(db_key, (num_vectors, vector_dim))
    
    # Partition the database along its first axis ('data')
    # The 8000 vectors are split evenly among 8 devices (1000 vectors per device).
    # JAX divides the array evenly along the sharded axis. Because 8000 % 8 == 0,
    # it splits perfectly. If it wasn't evenly divisible, JAX would raise an error.
    # The 128 dimensions are fully present on every device.
    db_sharding = NamedSharding(mesh, PartitionSpec('data', None))
    sharded_db = jax.device_put(mock_db, db_sharding)
    
    print("\nDatabase Sharding Layout:")
    jax.debug.visualize_array_sharding(sharded_db)
    # Expected Visualization Output:
    # Because it is sharded along the first axis (rows), you will see 8 horizontal chunks stacked vertically:
    # ┌───────────────────────────┐
    # │  CPU 0: db[0:1000, :]     │
    # ├───────────────────────────┤
    # │  CPU 1: db[1000:2000, :]  │
    # ├───────────────────────────┤
    # │  CPU 2: db[2000:3000, :]  │
    # ├───────────────────────────┤
    # │  CPU 3: db[3000:4000, :]  │
    # ├───────────────────────────┤
    # │  CPU 4: db[4000:5000, :]  │
    # ├───────────────────────────┤
    # │  CPU 5: db[5000:6000, :]  │
    # ├───────────────────────────┤
    # │  CPU 6: db[6000:7000, :]  │
    # ├───────────────────────────┤
    # │  CPU 7: db[7000:8000, :]  │
    # └───────────────────────────┘
    
    # Generate a batch of queries
    num_queries = 32
    mock_queries = jax.random.normal(query_key, (num_queries, vector_dim))
    
    # Replicate queries across all devices (no partitioning)
    # The entire batch of 32 queries is copied identically to all 8 devices.
    query_sharding = NamedSharding(mesh, PartitionSpec(None, None))
    sharded_queries = jax.device_put(mock_queries, query_sharding)
    
    print("\nQueries Sharding Layout:")
    jax.debug.visualize_array_sharding(sharded_queries)
    # Expected Visualization Output:
    # Because PartitionSpec(None, None) means the data is perfectly replicated,
    # you will see a single block representing the array with all CPUs sharing it:
    # ┌─────────────────────────────────────────────────────────────┐
    # │ CPU 0, CPU 1, ..., CPU 7: All devices hold queries[0:32, :] │
    # └─────────────────────────────────────────────────────────────┘

if __name__ == "__main__":
    main()
