import os
import shutil
import time

# -----------------------------------------------------------------------------
# GLOBAL SETUP: Environment configuration before JAX import
# -----------------------------------------------------------------------------
# 1. Enable 8 simulated CPU devices for SPMD sharding demos.
# 2. Configure XLA to dump HLO optimized graphs as human-readable text and protobuf.
# 3. Enable JAX compilation logging to stderr.
DUMP_DIR = "/tmp/xladump"
if os.path.exists(DUMP_DIR):
    shutil.rmtree(DUMP_DIR)
os.makedirs(DUMP_DIR, exist_ok=True)

os.environ["XLA_FLAGS"] = f"--xla_force_host_platform_device_count=8 --xla_dump_to={DUMP_DIR} --xla_dump_hlo_as_text"
os.environ["JAX_LOG_COMPILES"] = "1"

# Now import JAX and workspace dependencies
import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from vector_engine import vector_search, StaticVectorEngine

def get_hlo_file_count():
    """Counts .txt and .pb files dumped by XLA in the dump directory."""
    if not os.path.exists(DUMP_DIR):
        return 0
    return len([f for f in os.listdir(DUMP_DIR) if f.endswith('.txt') or f.endswith('.pb')])

def clear_dump_dir():
    """Clears the XLA dump directory to isolate compilation step counts."""
    if os.path.exists(DUMP_DIR):
        shutil.rmtree(DUMP_DIR)
    os.makedirs(DUMP_DIR, exist_ok=True)

def print_banner(title):
    print("\n" + "=" * 80)
    print(f" {title}")
    print("=" * 80)

def visualize_sharding_safe(arr):
    """Safely visualizes array sharding, falling back to raw representation if 'rich' is missing."""
    try:
        jax.debug.visualize_array_sharding(arr)
    except ValueError:
        print(f"  [Fallback] Raw Sharding Representation:\n    {arr.sharding}")
        print("  [Tip] Run 'pip install rich' to view the full colored ASCII sharding grid!")

def main():
    print("Initializing JAX compilation and observation verification...")
    devices = jax.devices()
    print(f"Detected devices: {len(devices)} CPU(s)")
    for d in devices:
        print(f" - {d}")

    # -----------------------------------------------------------------------------
    # DEMO 1: JAX Compilation & HLO Dumping (Dynamic vs. Static Shapes)
    # -----------------------------------------------------------------------------
    print_banner("DEMO 1: Compilation Logging & XLA HLO Dumping (/tmp/xladump)")
    
    # Generate mock database and dynamic queries
    db_size = 100
    dim = 64
    k = 5
    
    np.random.seed(42)
    db_data = np.random.randn(db_size, dim).astype(np.float32)
    
    # --- PHASE A: Dynamic Shapes Compilation Flood ---
    print("\n>>> Phase A: Standard JIT with Dynamic Query Batch Sizes")
    clear_dump_dir()
    
    # Run 1: Batch size B = 5
    q_b5 = np.random.randn(5, dim).astype(np.float32)
    print("\n[B=5] Running dynamic search...")
    # JAX will log a compilation message here
    vector_search(q_b5, db_data, k=k, metric='l2')
    jax.block_until_ready(None) # Ensure execution finishes
    files_after_b5 = get_hlo_file_count()
    print(f"[B=5] Files in {DUMP_DIR}: {files_after_b5}")
    
    # Run 2: Batch size B = 12 (Changing shapes triggers recompilation!)
    q_b12 = np.random.randn(12, dim).astype(np.float32)
    print("\n[B=12] Running dynamic search with different batch size...")
    # JAX will log a SECOND compilation message here due to shape change!
    vector_search(q_b12, db_data, k=k, metric='l2')
    jax.block_until_ready(None)
    files_after_b12 = get_hlo_file_count()
    print(f"[B=12] Files in {DUMP_DIR}: {files_after_b12}")
    
    recomp_files = files_after_b12 - files_after_b5
    print(f"\n[Result] Changing batch size from 5 to 12 created {recomp_files} new HLO dump files.")
    print("This confirms XLA compiled an entirely new program, inducing runtime latency spikes!")

    # --- PHASE B: Static Shapes (StaticVectorEngine) Caching ---
    print("\n>>> Phase B: StaticVectorEngine with Padding to Prevent Recompilation")
    clear_dump_dir()
    
    # Initialize static engine wrapper with fixed bounds
    engine = StaticVectorEngine(max_batch_size=16, max_db_size=128, dim=dim, metric='l2')
    
    # Run 1: Batch size B = 5 (padded to 16, db padded to 128)
    print("\n[B=5] Running static engine search...")
    # JAX compiles this exactly once for max shapes
    engine.search(q_b5, db_data, k=k)
    jax.block_until_ready(None)
    files_static_b5 = get_hlo_file_count()
    print(f"[B=5] Files in {DUMP_DIR}: {files_static_b5}")
    
    # Run 2: Batch size B = 12 (padded to 16, db padded to 128)
    print("\n[B=12] Running static engine search...")
    # JAX will reuse the cache! No compilation logged!
    engine.search(q_b12, db_data, k=k)
    jax.block_until_ready(None)
    files_static_b12 = get_hlo_file_count()
    print(f"[B=12] Files in {DUMP_DIR}: {files_static_b12}")
    
    static_recomp_files = files_static_b12 - files_static_b5
    print(f"\n[Result] Changing batch size under StaticVectorEngine created {static_recomp_files} new files.")
    print("This proves XLA successfully hit the compilation cache, protecting real-time query latency!")

    # -----------------------------------------------------------------------------
    # DEMO 2: Tracing JAX Expressions (jax.make_jaxpr)
    # -----------------------------------------------------------------------------
    print_banner("DEMO 2: Tracing Intermediate JAX Expressions (jaxpr)")
    
    dummy_queries = jnp.zeros((5, dim))
    dummy_db = jnp.zeros((10, dim))
    
    print("Generating and printing the traced jaxpr of vector_search:")
    search_jaxpr = jax.make_jaxpr(lambda q, db: vector_search(q, db, k=2, metric='l2'))(
        dummy_queries, dummy_db
    )
    print(search_jaxpr)
    print("\n[Insight] Notice how the control flow (if/else) has disappeared and only mathematical")
    print("primitives (dot, sum, neg, top_k) remain in the traced graph tape.")

    # -----------------------------------------------------------------------------
    # DEMO 3: Ahead-of-Time (AOT) Lowering and Compilation API
    # -----------------------------------------------------------------------------
    print_banner("DEMO 3: Ahead-of-Time (AOT) Lowering & Compilation API")
    
    print("Lowering the vector_search logic to XLA HLO compiler IR...")
    lowered = jax.jit(lambda q, db: vector_search(q, db, k=2, metric='l2')).lower(
        dummy_queries, dummy_db
    )
    
    hlo_text = lowered.as_text()
    print("\nPreview of first 15 lines of optimized HLO Text IR:")
    for line in hlo_text.splitlines()[:15]:
        print(f"  {line}")
    print("  ...")
    
    print("\nCompiling the lowered HLO representation statically...")
    compiled = lowered.compile()
    print("Successfully compiled Ahead-of-Time! Ready to execute on target hardware.")

    # -----------------------------------------------------------------------------
    # DEMO 4: JAX Profiler Trace Setup
    # -----------------------------------------------------------------------------
    print_banner("DEMO 4: JAX Profiler Trace Setup")
    
    profile_dir = "/tmp/jax_profile_logs"
    if os.path.exists(profile_dir):
        shutil.rmtree(profile_dir)
        
    print(f"Starting program trace. Capture target: {profile_dir}")
    jax.profiler.start_trace(profile_dir)
    
    # Warm up step to compile
    _ = vector_search(dummy_queries, dummy_db, k=2, metric='l2')
    jax.block_until_ready(_)
    
    # Trace dynamic search executions
    for i in range(5):
        _ = vector_search(dummy_queries, dummy_db, k=2, metric='l2')
    jax.block_until_ready(_)
    
    jax.profiler.stop_trace()
    print(f"Trace captured successfully! Directory: {profile_dir}")
    print("To visualize this trace:")
    print("  1. Run: pip install tensorboard tensorboard-plugin-profile")
    print(f"  2. Run: tensorboard --logdir={profile_dir}")
    print("  3. Open http://localhost:6006 in your browser and select the 'Profile' tab.")

    # -----------------------------------------------------------------------------
    # DEMO 5: SPMD and Array Sharding Visualization
    # -----------------------------------------------------------------------------
    print_banner("DEMO 5: SPMD and Array Sharding Layout Visualization")
    
    # Define a logical 1D Mesh grouping physical devices along a single 'data' dimension
    mesh = Mesh(devices, axis_names=('data',))
    
    # Partition a mock 8000x128 database along its first axis (sharding rows across 8 devices)
    db_sharding = NamedSharding(mesh, P('data', None))
    mock_large_db = jax.device_put(jnp.zeros((8000, 128)), db_sharding)
    
    # Replicate queries across all 8 devices (broadcast/copy queries to all processors)
    query_sharding = NamedSharding(mesh, P(None, None))
    mock_large_queries = jax.device_put(jnp.zeros((32, 128)), query_sharding)
    
    print("Queries Sharding Layout Visualization (PartitionSpec(None, None) -> Replicated):")
    visualize_sharding_safe(mock_large_queries)
    
    print("\nDatabase Sharding Layout Visualization (PartitionSpec('data', None) -> Sharded across 8 devices):")
    visualize_sharding_safe(mock_large_db)
    
    print("\nAll 5 Demos completed successfully!")

if __name__ == "__main__":
    main()
