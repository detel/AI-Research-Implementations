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
