# Memory Management in RAPIDS with RMM

RMM stands for RAPIDS Memory Manager

What RMM is:

 - Replacement allocator for CUDA Device Memory
 - Pool allocator to make CUDA device memory allocation / deallocation faster
 - Central place for all device memory allocations in cuDF (libGDF & pyGDF)

What RMM is not (at least today):
 - Replacement allocator for CUDA managed memory (Unified Memory, 
   e.g. `cudaMallocManaged`)
 - Replacement allocator for host memory (`malloc`, `new`, `cudaMallocHost`, 
   `cudaHostRegister`)

## Using RMM in C/C++ code

Using RMM in CUDA C++ code is straightforward. Replace calls to `cudaMalloc()` 
with calls to `rmmAlloc()`, and calls to `cudaFree()` with calls to `rmmFree()`.
Note that `rmmAlloc` and `rmmFree` take an additional parameter, a stream 
identifier. These are necessary to enable asynchronous allocation and 
deallocation, however, the default (also known as null) stream (or `0`) can be
used. For example:

```
// old
CUDA_TRY( cudaMalloc(&myvar, size_in_bytes) );
// ...
CUDA_TRY( cudaFree(myvar) );

// new
RMM_TRY( rmmAlloc((void**)&myvar, size_in_bytes, stream_id) );
// ...
RMM_TRY( rmmFree(myvar, stream_id) );
```

`RMM_TRY` is a macro equivalent to `CUDA_TRY`. It returns a GDF error code if 
the RMM function fails.

### Using RMM with Thrust

libGDF makes heavy use of Thrust. Thrust uses CUDA device memory in two 
situations:

 1. As the backing store for `thrust::device_vector`, and
 2. As temporary storage inside some algorithms, such as `thrust::sort`.

libGDF now includes a custom Thrust allocator in the file 
`thrust_rmm_allocator.h`. This defines the template class `rmm_allocator`, and 
an alias for algorithm temporary storage called `rmm_temp_allocator`. 

#### Thrust Device Vectors

Instead of creating device vectors like this:

```
thrust::device_vector<size_type> permuted_indices(column_length);
```

You can tell Thrust to use `rmm_allocator` like this:

```
thrust::device_vector<size_type, rmm_allocator<T>> permuted_indices(column_length);
```

For convenience, usually you will want to create an alias, like this:

```template <typename T> 
using Vector = thrust::device_vector<T, rmm_allocator<T>>;

...

Vector<size_type> permuted_indices(column_length);
```

(TODO: add a definition of this alias in an include so all files can easily use 
it.)

#### Thrust Algorithms

To instruct Thrust to use RMM to allocate temporary storage, you need to create
an execution policy that uses it, like this:

```
rmm_temp_allocator allocator(stream);

thrust::sort(thrust::cuda::par(allocator).on(stream), ...);
```

Note a current Thrust bug prevents this from being a one-liner. 

(TODO: define the execution policy in an include so all files can easily use it.)

## Using RMM in Python Code

Pygdf and other Python libraries typically create arrays of CUDA device memory
by using Numba's `cuda.device_array` interfaces. Until Numba provides a plugin
interface for using an external memory manager, RMM provides an API compatible
with `cuda.device_array` constructors that Pygdf (and Libgdf pytests) should use
to ensure all CUDA device memory is allocated via the memory manager. RMM 
provides:

   - `librmm.device_array()`
   - `librmm.device_array_like()`
   - `librmm.to_device()`
   - `librmm.auto_device()`
   
Which are compatible with their Numba `cuda.*` equivalents. They return a Numba 
NDArray object whose memory is allocated in CUDA device memory using RMM.

`librmm` also provides `get_ipc_handle()` for getting the IPC handle associated 
with a Numba NDArray, which accounts for the case where the data for the NDArray
is suballocated from some larger pool allocation by the memory manager.

To use librmm NDArray functions you need to import librmm like this:

`from librmm_cffi import librmm` or
`from librmm_cffi import librmm as rmm`

### Handling RMM Options in Python Code

RMM currently defaults to just calling cudaMalloc, but you can enable the 
experimental pool allocator using the `librmm_config` module. 

```
from librmm_cffi import librmm_config as rmm_cfg

rmm_cfg.use_pool_allocator = True # default is False
rmm_cfg.initial_pool_size = 2<<30 # set to 2GiB. Default is 1/2 total GPU memory
rmm_cfg.enable_logging = True # default is False -- has perf overhead
```

To configure RMM options to be used in Pygdf before loading, simply do the above 
before you `import pygdf`. You can re-initialize the memory manager with different 
settings at run time by calling `librmm.finalize()`, then changing the above 
options, and then calling `librmm.initialize()`.

You can also optionally use the internal functions in PyGDF which call these 
functions. Here are some example configuration functions that can be used in 
a notebook to initialize the memory manager in each Dask worker.

```
from librmm_cffi import librmm_config as rmm_cfg

def initialize_rmm_pool():
    pygdf._gdf.rmm_finalize()
    rmm_cfg.use_pool_allocator = True
    return pygdf._gdf.rmm_initialize()

def initialize_rmm_no_pool():
    pygdf._gdf.rmm_finalize()
    rmm_cfg.use_pool_allocator = False
    return pygdf._gdf.rmm_initialize()

def finalize_rmm():
    return pygdf._gdf.rmm_finalize()
```

Given the above, typically you would initialize RMM in the notebook process to
not use the pool `initialize_rmm_no_pool()`, and then run 
`client.run(initialize_rmm_pool) to initialize a memory pool in each worker
process.

Remember that while the pool is in use memory is not freed. So if you follow 
Pygdf operations with device-memory-intensive computations that don't use RMM
(such as XGBoost), you will need to move the data to the host and then 
finalize RMM. The Mortgage E2E workflow notebook uses this technique. We are 
working on better ways to reclaim memory, as well as making RAPIDS machine
learning libraries use the same RMM memory pool.
