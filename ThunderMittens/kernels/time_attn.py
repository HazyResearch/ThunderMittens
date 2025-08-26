import mlx.core as mx
from tk import attn_fwd

import time
def calculate_flops(b, h, n, d):
    # Calculate the total flops for attention
    attn_flops = (
        2 * b * h * n * n * d +   # Q * K^T: 2BHNND (multiply-add)
        4 * b * h * n * n +       # Softmax: 2BHNN (exp and divide, plus flash-attn bookkeeping)
        2 * b * h * n * n * d     # (Q * K^T) * V: 2BHNND (multiply-add)
    )
    return attn_flops / 1e9  # Convert to GFLOPs

def run_mlx_iterations(q, k, v,itt):
    for _ in range(itt):
        mx.eval(mx.fast.scaled_dot_product_attention(
            q, k, v, scale=1, mask=None
        ))
    toi = time.perf_counter()
    for _ in range(itt):
        mx.eval(mx.fast.scaled_dot_product_attention(q, k, v, scale=1, mask=None))
    toc = time.perf_counter()
    tpi = 1e3 * (toc - toi) / itt
    return tpi

def run_tk_iterations(q, k, v, itt):
    for _ in range(itt):
        mx.eval(mx.fast.scaled_dot_product_attention(
            q, k, v, scale=1, mask=None
        ))
    toi = time.perf_counter()
    for _ in range(itt):
        mx.eval(attn_fwd(q, k, v))
    toc = time.perf_counter()
    tpi = 1e3 * (toc - toi) / itt
    return tpi


def benchmark_sdpa(q: mx.array, k: mx.array, v: mx.array, itt):
    mlx_tpi = run_mlx_iterations(q, k, v, itt)

    gflops_mlx = calculate_flops(q.shape[0], q.shape[1], q.shape[2], q.shape[3]) / (mlx_tpi / 1000)
    print(f"mlx: {gflops_mlx:.2f} GFLOPS")

    tk_tpi  = run_tk_iterations(q, k, v, itt)
    gflops_tk = calculate_flops(q.shape[0], q.shape[1], q.shape[2], q.shape[3]) / (tk_tpi / 1000)
    print(f"tk : {gflops_tk:.2f} GFLOPS")
    print(f"diff: {gflops_tk / gflops_mlx*100}%")
    print("---")

for D in [64, 128]:
    for N in [512*1, 512*2, 512*3, 512*4, 512*5]:
        q = mx.random.uniform(shape=(16, 16, N, D)).astype(mx.bfloat16)
        k = mx.random.uniform(shape=(16, 16, N, D)).astype(mx.bfloat16)
        v = mx.random.uniform(shape=(16, 16, N, D)).astype(mx.bfloat16)
        itt = 50
        for i in range(1):
            mx.metal.clear_cache()
            print(f"running: ({q.shape[0]} x {q.shape[1]} x {q.shape[2]} x {q.shape[3]}), {itt} warmups, {itt} benchmarked")
            benchmark_sdpa(q, k, v, itt)
    print("------------------------------------------------------------------------------")
