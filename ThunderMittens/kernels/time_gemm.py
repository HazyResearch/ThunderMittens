import mlx.core as mx
from tk import matmul_custom

import time
def calculate_flops(n, m, k):
    # Calculate the total flops for attention
    attn_flops = (
        2 * n * m * k
    )
    return attn_flops / 1e9  # Convert to GFLOPs

def run_mlx_iterations(a,b,itt):
    for _ in range(itt):
        mx.eval(mx.matmul(a,b,stream=mx.gpu))  
    toi = time.perf_counter()
    for _ in range(itt):
        mx.eval(mx.matmul(a,b,stream=mx.gpu))
    toc = time.perf_counter()
    tpi = 1e3 * (toc - toi) / itt
    return tpi

def run_tk_iterations(a,b,itt):
    for _ in range(itt):
        mx.eval(matmul_custom(a,b,stream=mx.gpu))  
    toi = time.perf_counter()
    for _ in range(itt):
        mx.eval(matmul_custom(a,b,stream=mx.gpu))
    toc = time.perf_counter()
    tpi = 1e3 * (toc - toi) / itt
    return tpi

def benchmark_sdpa(a: mx.array, b: mx.array, itt):
    mlx_tpi = run_mlx_iterations(a,b, itt)
    gflops_mlx = calculate_flops(a.shape[0], a.shape[1], b.shape[1]) / (mlx_tpi / 1000)
    print(f"mlx: {gflops_mlx:.2f} GFLOPS")

    tk_tpi = run_tk_iterations(a,b, itt)
    gflops_tk = calculate_flops(a.shape[0], a.shape[1], b.shape[1]) / (tk_tpi / 1000)
    print(f"tk: {gflops_tk:.2f} GFLOPS")


for NKM in [1024*1, 1024*2, 1024*4, 1024*8, 1024*16]:
    N = NKM
    K = NKM
    M = NKM
    a = mx.random.uniform(shape=(N, K)).astype(mx.bfloat16)
    b = mx.random.uniform(shape=(K, M)).astype(mx.bfloat16)
    itt = int(1024*200 / NKM)
    for i in range(3):
        mx.metal.clear_cache()
        print(f"running: ({a.shape[0]} x {b.shape[1]} x {a.shape[1]}), {itt} warmups, {itt} benchmarked")
        benchmark_sdpa(a,b, itt) 
    print("------------------------------------------------------------------------------")

a = mx.ones(shape=(32, 16)).astype(mx.float32)
b = mx.ones(shape=(16, 32)).astype(mx.float32)


mx.random.seed(42)
