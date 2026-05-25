import os
os.environ["XLA_FLAGS"] = '--xla_force_host_platform_device_count=8'
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vector_engine import (
    l2_distance,
    cosine_distance,
    dot_product_distance,
    vector_search
)

@pytest.fixture
def sample_data():
    queries = np.array([
        [1.0, 0.0],
        [0.0, 2.0]
    ], dtype=np.float32)
    
    database = np.array([
        [1.0, 0.0],   # Exact match for query 0
        [0.0, 1.0],   # Orthogonal to query 0, same direction as query 1
        [-1.0, 0.0],  # Opposite direction to query 0
        [0.0, 2.0]    # Exact match for query 1
    ], dtype=np.float32)
    
    return queries, database

def test_l2_distance(sample_data):
    queries, database = sample_data
    
    dist = l2_distance(jnp.array(queries), jnp.array(database))
    
    # Expected L2 distances squared
    # q0: [1,0] vs [[1,0], [0,1], [-1,0], [0,2]] -> [0, 2, 4, 5]
    # q1: [0,2] vs [[1,0], [0,1], [-1,0], [0,2]] -> [5, 1, 5, 0]
    expected = np.array([
        [0.0, 2.0, 4.0, 5.0],
        [5.0, 1.0, 5.0, 0.0]
    ])
    
    np.testing.assert_allclose(np.array(dist), expected, atol=1e-5)

def test_cosine_distance(sample_data):
    queries, database = sample_data
    
    dist = cosine_distance(jnp.array(queries), jnp.array(database))
    
    # Cosine distances (1 - similarity)
    # q0: [1,0] vs database
    # sim q0: [1.0, 0.0, -1.0, 0.0]
    # dist q0: [0.0, 1.0, 2.0, 1.0]
    
    # q1: [0,2] vs database
    # sim q1: [0.0, 1.0, 0.0, 1.0]
    # dist q1: [1.0, 0.0, 1.0, 0.0]
    
    expected = np.array([
        [0.0, 1.0, 2.0, 1.0],
        [1.0, 0.0, 1.0, 0.0]
    ])
    
    np.testing.assert_allclose(np.array(dist), expected, atol=1e-5)

def test_dot_product_distance(sample_data):
    queries, database = sample_data
    
    dist = dot_product_distance(jnp.array(queries), jnp.array(database))
    
    # Dot products
    # q0: [1,0] vs [[1,0], [0,1], [-1,0], [0,2]] -> [1.0, 0.0, -1.0, 0.0]
    # q1: [0,2] vs [[1,0], [0,1], [-1,0], [0,2]] -> [0.0, 2.0, 0.0, 4.0]
    
    expected = np.array([
        [1.0, 0.0, -1.0, 0.0],
        [0.0, 2.0, 0.0, 4.0]
    ])
    
    np.testing.assert_allclose(np.array(dist), expected, atol=1e-5)

def test_vector_search_l2(sample_data):
    queries, database = sample_data
    k = 2
    
    dists, indices = vector_search(jnp.array(queries), jnp.array(database), k=k, metric='l2')
    
    # L2 distances
    # q0 distances: [0, 2, 4, 5] -> top 2 smallest: 0.0 (idx 0), 2.0 (idx 1)
    # q1 distances: [5, 1, 5, 0] -> top 2 smallest: 0.0 (idx 3), 1.0 (idx 1)
    
    expected_dists = np.array([
        [0.0, 2.0],
        [0.0, 1.0]
    ])
    expected_indices = np.array([
        [0, 1],
        [3, 1]
    ])
    
    np.testing.assert_allclose(np.array(dists), expected_dists, atol=1e-5)
    np.testing.assert_array_equal(np.array(indices), expected_indices)

def test_vector_search_dot_product(sample_data):
    queries, database = sample_data
    k = 2
    
    dists, indices = vector_search(jnp.array(queries), jnp.array(database), k=k, metric='dot_product')
    
    # Dot products
    # q0 dot products: [1, 0, -1, 0] -> top 2 largest: 1.0 (idx 0), 0.0 (idx 1 or 3)
    # q1 dot products: [0, 2, 0, 4] -> top 2 largest: 4.0 (idx 3), 2.0 (idx 1)
    
    np.testing.assert_allclose(np.array(dists[0, 0]), 1.0, atol=1e-5)
    assert indices[0, 0] == 0
    np.testing.assert_allclose(np.array(dists[0, 1]), 0.0, atol=1e-5)
    
    np.testing.assert_allclose(np.array(dists[1]), np.array([4.0, 2.0]), atol=1e-5)
    np.testing.assert_array_equal(np.array(indices[1]), np.array([3, 1]))

def test_invalid_metric():
    queries = jnp.ones((2, 4))
    database = jnp.ones((5, 4))
    
    with pytest.raises(ValueError, match="Unknown metric: invalid"):
        vector_search(queries, database, k=2, metric='invalid')

def test_distributed_vector_search():
    from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
    from vector_engine import distributed_vector_search
    
    # Verify devices
    devices = jax.devices()
    assert len(devices) == 8, "Expected 8 simulated CPU devices"
    
    # Create mesh
    mesh = Mesh(devices, axis_names=('data',))
    
    # Generate random test data
    key = jax.random.PRNGKey(42)
    db_key, query_key = jax.random.split(key)
    
    N, D, B, K = 8000, 128, 32, 10
    
    mock_db = jax.random.normal(db_key, (N, D))
    mock_queries = jax.random.normal(query_key, (B, D))
    
    # Shard database and replicate queries
    db_sharding = NamedSharding(mesh, P('data', None))
    query_sharding = NamedSharding(mesh, P(None, None))
    
    sharded_db = jax.device_put(mock_db, db_sharding)
    sharded_queries = jax.device_put(mock_queries, query_sharding)
    
    # Run regular vector search (baseline)
    expected_dists, expected_indices = vector_search(mock_queries, mock_db, k=K, metric='l2')
    
    # Run distributed vector search
    dist_dists, dist_indices = distributed_vector_search(mesh, sharded_queries, sharded_db, k=K, metric='l2')
    
    # Assert exact match
    np.testing.assert_allclose(dist_dists, expected_dists, atol=1e-5)
    np.testing.assert_array_equal(dist_indices, expected_indices)

def test_static_vector_engine():
    from vector_engine import StaticVectorEngine
    
    # Initialize the static engine wrapper
    max_b = 10
    max_n = 20
    dim = 4
    k = 2
    
    engine = StaticVectorEngine(max_batch_size=max_b, max_db_size=max_n, dim=dim, metric='l2')
    
    # Generate dynamic data smaller than max bounds
    B = 3
    N = 5
    
    key = jax.random.PRNGKey(99)
    db_key, query_key = jax.random.split(key)
    
    # Use numpy for the host-side inputs
    queries = np.array(jax.random.normal(query_key, (B, dim)))
    database = np.array(jax.random.normal(db_key, (N, dim)))
    
    # Run the unpadded baseline
    expected_dists, expected_indices = vector_search(queries, database, k=k, metric='l2')
    
    # Run the padded engine
    padded_dists, padded_indices = engine.search(queries, database, k=k)
    
    # Check that they match exactly, proving the padded zero-vectors didn't pollute the top-K
    np.testing.assert_allclose(padded_dists, expected_dists, atol=1e-5)
    np.testing.assert_array_equal(padded_indices, expected_indices)
