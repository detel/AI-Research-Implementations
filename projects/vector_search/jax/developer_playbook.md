# JAX/XLA Developer Playbook: Observation, Profiling & Debugging Handbook

This handbook provides step-by-step instructions on how to run, analyze, and troubleshoot JAX/XLA compilation and execution behavior using advanced observation tools. Use this playbook alongside the verification script to monitor performance, debug sharding layout issues, and prevent latency spikes.

---

## 1. Verifying Code Correctness & Functional Tests

Before profiling performance, always verify that your mathematical and distributed logic is functionally correct.

### Running Unit Tests
Unit tests use `pytest` and mock 8 CPU devices to verify the correctness of the distance metrics, the manual SPMD sharded search, and the static padding engine.

```bash
# Execute the test suite
pytest projects/vector_search/jax/test_vector_engine.py
```

---

## 2. JAX Compilation Logging & HLO Dumping

XLA compiles JAX graphs into optimized high-level optimizer (HLO) programs. 

### Step 1: Running with Compilation Logging
Set `JAX_LOG_COMPILES=1` to print compiler events directly to stderr. This is the fastest way to detect if dynamic shapes are leaking into your hot paths.

```bash
JAX_LOG_COMPILES=1 python projects/vector_search/jax/verify_compilation.py
```
* **What to observe**: Look at the terminal output. If you see lines starting with `[JAX] Compiling ...` during evaluation or search iterations, recompilation is occurring, destroying real-time latency.

### Step 2: Extracting HLO Dumps
To write the actual intermediate compiler graphs to disk:
1. Ensure the dump directory exists:
   ```bash
   mkdir -p /tmp/xladump
   ```
2. Run your JAX script with `XLA_FLAGS` pointing to the dump location:
   ```bash
   XLA_FLAGS="--xla_dump_to=/tmp/xladump --xla_dump_hlo_as_text" python projects/vector_search/jax/verify_compilation.py
   ```
3. Navigate to `/tmp/xladump` to view the generated files. You will see:
   * **`.txt` files**: Human-readable text representations of the HLO IR.
   * **`.pb` files**: Binary protobuf serialization of the compiler graphs.

### Step 3: Reading HLO Code
Open the HLO `.txt` files in a text editor. Search for the following key terms to analyze compiler decisions:
* **`%fusion`**: XLA groups multiple elementwise operations (like add, subtract, multiply) into a single loop to avoid reading/writing to device memory. Look for fused operators to verify efficient execution.
* **`dot`**: Represents matrix multiplications (such as `jnp.dot` in our distance calculations).
* **`reduce` / `reduce-window`**: Represents reduction passes (such as `jnp.sum` or `jax.lax.top_k`).

---

## 3. Tracing JAX Expressions via `jaxpr`

A `jaxpr` (JAX expression) is JAX's internal representation of the traced Python function *before* it is handed over to XLA.

### Inspecting the trace
1. Use `jax.make_jaxpr` inside your code or interactive Python environment:
   ```python
   import jax
   from vector_engine import l2_distance
   import jax.numpy as jnp

   queries = jnp.ones((5, 128))
   db = jnp.ones((100, 128))
   
   jaxpr = jax.make_jaxpr(l2_distance)(queries, db)
   print(jaxpr)
   ```
2. **What to observe**: JAX prints a highly structured, single-static-assignment (SSA) tape of operations:
   * `let` bindings assigning outputs of primitive JAX ops (e.g., `add`, `mul`, `dot`).
   * **Tracer Variables**: Dynamic array values appear as tracers (e.g., `a:f32[5,128]`).
   * **Static Constants**: If a parameter was JIT-compiled as static (`static_argnames`), its values are hardcoded directly into the instructions rather than appearing as variables.

---

## 4. Ahead-of-Time (AOT) Lowering and Inspecting

The AOT API allows you to inspect optimized compiler graphs statically without running any execution phases.

### Inspecting lowered stages
Use the `.lower()` and `.compile()` APIs on your jitted functions:
```python
import jax
from vector_engine import vector_search
import jax.numpy as jnp

# 1. Trace and lower to HLO stage
lowered = jax.jit(vector_search, static_argnames=['k', 'metric']).lower(
    jnp.ones((5, 128)), jnp.ones((100, 128)), k=5, metric='l2'
)

# 2. Print the optimized HLO text representation
print(lowered.as_hlo_text())

# 3. Perform final machine code compilation
compiled = lowered.compile()
```
* **Use Case**: This is incredibly useful for writing performance-critical library code, verifying that your graph structure is mathematically minimized before executing it on target devices.

---

## 5. Capturing and Opening the TensorBoard Profiler

The TensorBoard Profiler provides a detailed visual timeline of your program's execution, separating host CPU processing, memory transfers, and device accelerator operations.

### Step 1: Capturing the Trace
You can capture a trace programmatically by wrapping execution sections inside JAX's profiling context:
```python
import jax.profiler

# Start writing tracing metrics to a local folder
jax.profiler.start_trace("/tmp/jax_profile_logs")

# Run warm-up step to compile the graph (profile compiles separately)
result = vector_search(queries, db, k=5)

# Profile actual execution steps
for _ in range(10):
    result = vector_search(queries, db, k=5)

# Finalize the profiling dump
jax.profiler.stop_trace()
```

### Step 2: Opening the Profiler in TensorBoard
1. Install TensorBoard and its specialized PyTorch/JAX profiling plugin:
   ```bash
   pip install tensorboard tensorboard-plugin-profile
   ```
2. Start the TensorBoard server, pointing it to your profile logs directory:
   ```bash
   tensorboard --logdir=/tmp/jax_profile_logs --port=6006
   ```
3. Open your web browser and navigate to:
   [http://localhost:6006](http://localhost:6006)
4. In the top-right navigation menu, select **Profile** from the dropdown list.

### Step 3: Analyzing the Profile
* **Trace Viewer Tab**: Visualizes a step-by-step Gantt chart of host-side execution vs. device-side execution. Look for large gaps where the accelerator is idle, indicating significant host overhead or compilation bottlenecks.
* **Memory Profile Tab**: Tracks memory allocation over time, letting you detect leaks and peak memory consumption to proactively prevent Out-of-Memory (OOM) errors.

---

## 6. Sharding and SPMD Visualization

When sharding massive vector databases across multiple GPUs or CPUs (SPMD distributed search), you must make sure memory layouts are optimal.

### Visualizing Layouts
In your distributed scripts, call `visualize_array_sharding` to inspect sharding layouts:
```python
import jax
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

devices = jax.devices()
mesh = Mesh(devices, axis_names=('data',))

# Replicated queries across all mesh devices
query_sharding = NamedSharding(mesh, P(None, None))
sharded_queries = jax.device_put(queries, query_sharding)

# Sharded database split along first axis
db_sharding = NamedSharding(mesh, P('data', None))
sharded_db = jax.device_put(database, db_sharding)

print("Queries Placement:")
jax.debug.visualize_array_sharding(sharded_queries)

print("\nDatabase Placement:")
jax.debug.visualize_array_sharding(sharded_db)
```

### Interpreting the Output
The utility prints ASCII grid layouts representing device allocations:
* **Single Solid Block (`CPU 0, CPU 1, ..., CPU 7`)**: Indicates data is replicated on all devices. Great for queries which must be fast and locally accessible to all devices.
* **Split Stacked Blocks (`CPU 0`, `CPU 1`, etc. stacked)**: Indicates the tensor is partitioned along that dimension. In this case, the vector database is split evenly among devices, meaning each device processes a subset of database rows concurrently.

---

## 7. Low-Level Accelerator Profiling (NVIDIA Nsight)

For bare-metal CUDA execution debugging on GPUs, systems engineers use NVIDIA Nsight tools.

### Step 1: Profiling Command
Profile your JAX Python script using Nsight Systems (`nsys`):
```bash
nsys profile -w true -t cuda,nvtx,osrt,opg -o /tmp/nsys_vector_profile python projects/vector_search/jax/verify_compilation.py
```
This generates an `/tmp/nsys_vector_profile.nsys-rep` report file.

### Step 2: Visualizing in Nsight GUI
1. Download and open the **NVIDIA Nsight Systems** desktop application.
2. Load the `.nsys-rep` report file.
3. Observe the timeline:
   * Look at **CUDA API Calls** and **GPU Kernels**.
   * Verify that XLA fuses your elementwise L2 distance computations into unified CUDA kernels rather than calling separate, individual kernels, which dramatically reduces memory bandwidth bottlenecks.
