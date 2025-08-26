
#include "../../../testing_commons/testing_flags.hpp"
#ifdef TEST_WARP_MEMORY_TILE_SHARED_TO_REGISTER
#include "../../../../../include/tk.metal"

namespace mittens {

#define gts_kern_PARAMS(T) \
    device T* _input [[buffer(0)]], \
    device T* _output [[buffer(1)]], \
    uint3         id              [[thread_position_in_grid]], \
    uint3         thread_group_id [[threadgroup_position_in_grid]], \
    uint          simd_group_id   [[simdgroup_index_in_threadgroup]], \
    uint          simd_lane_id    [[thread_index_in_simdgroup]]
constant constexpr int B = 3, D = 1, R = 4, C = 5;
template<typename T, unsigned H, unsigned W, typename L>
kernel void gts(gts_kern_PARAMS(T))
{
    using reg_tile = rt<T, H * 8, W * 8, L>;
    using smem_tile = st<T, H * 8, W * 8>;
    using GL = gl<T, B, D, 16*R*H, 8*C*W>;
    threadgroup smem_tile smem_A, smem_B;
    reg_tile reg_A;
    GL input(_input, nullptr, nullptr, nullptr, nullptr);
    GL output(_output, nullptr, nullptr, nullptr, nullptr);
    for(int i = 0; i < input.batch; i++)
    for(int j = 0; j < input.depth; j++)
    for(int k = 0; k < input.rows/smem_tile::rows; k++)
    for(int l = 0; l < input.cols/smem_tile::cols; l++) {
        load(smem_A, input, {i, j, k, l}, simd_lane_id);
        metal::simdgroup_barrier(metal::mem_flags::mem_none);
        load(reg_A, smem_A, simd_lane_id);
        store(smem_B, reg_A, simd_lane_id);
        metal::simdgroup_barrier(metal::mem_flags::mem_none);
        store(output, smem_B, {i, j, k, l}, simd_lane_id);
    }
}

#define gen_kernel(reg_N, reg_M) \
    template [[host_name("shared_reg_loadstore_gmem_float_" #reg_N "x" #reg_M "_rt_row_layout")]] [[kernel]] \
    void gts<float, reg_N, reg_M, ducks::rt_layout::row>(gts_kern_PARAMS(float)); \
    template [[host_name("shared_reg_loadstore_gmem_float_" #reg_N "x" #reg_M "_rt_col_layout")]] [[kernel]] \
    void gts<float, reg_N, reg_M, ducks::rt_layout::col>(gts_kern_PARAMS(float)); \
    template [[host_name("shared_reg_loadstore_gmem_bf16_" #reg_N "x" #reg_M "_rt_row_layout")]] [[kernel]] \
    void gts<bf16, reg_N, reg_M, ducks::rt_layout::row>(gts_kern_PARAMS(bf16)); \
    template [[host_name("shared_reg_loadstore_gmem_bf16_" #reg_N "x" #reg_M "_rt_col_layout")]] [[kernel]] \
    void gts<bf16, reg_N, reg_M, ducks::rt_layout::col>(gts_kern_PARAMS(bf16)); \
    template [[host_name("shared_reg_loadstore_gmem_half_" #reg_N "x" #reg_M "_rt_row_layout")]] [[kernel]] \
    void gts<half, reg_N, reg_M, ducks::rt_layout::row>(gts_kern_PARAMS(half)); \
    template [[host_name("shared_reg_loadstore_gmem_half_" #reg_N "x" #reg_M "_rt_col_layout")]] [[kernel]] \
    void gts<half, reg_N, reg_M, ducks::rt_layout::col>(gts_kern_PARAMS(half));
    
    //ducks::rt_layout::row
    
#if (INTENSITY_1)
    #define gen_kernels(N) \
        gen_kernel(N,1)  gen_kernel(N,2)

    gen_kernels(1) gen_kernels(2)
    #undef gen_kernels
#elif (INTENSITY_2)
    #define gen_kernels(N) \
        gen_kernel(N,1)  gen_kernel(N,2)  gen_kernel(N,3)  gen_kernel(N,4)

    gen_kernels(1) gen_kernels(2) gen_kernels(3) gen_kernels(4)
    #undef gen_kernels
#elif (INTENSITY_3) // 8
    
    #define gen_kernels(N) \
        gen_kernel(N,1)  gen_kernel(N,2)  gen_kernel(N,3)  gen_kernel(N,4)  gen_kernel(N,5)  gen_kernel(N,6)  gen_kernel(N,7)  gen_kernel(N,8)

    gen_kernels(1) gen_kernels(2) gen_kernels(3) gen_kernels(4) gen_kernels(5) gen_kernels(6) gen_kernels(7) gen_kernels(8)
    #undef gen_kernels
#elif (INTENSITY_4) // 16
    #define gen_kernels(N) \
        gen_kernel(N,1)  gen_kernel(N,2)  gen_kernel(N,3)  gen_kernel(N,4)  gen_kernel(N,5)  gen_kernel(N,6)  gen_kernel(N,7)  gen_kernel(N,8)  \
        gen_kernel(N,9)  gen_kernel(N,10) gen_kernel(N,11) gen_kernel(N,12) gen_kernel(N,13) gen_kernel(N,14) gen_kernel(N,15) gen_kernel(N,16)

    gen_kernels(1) gen_kernels(2) gen_kernels(3) gen_kernels(4) gen_kernels(5) gen_kernels(6) gen_kernels(7) gen_kernels(8)
    gen_kernels(9) gen_kernels(10) gen_kernels(11) gen_kernels(12) gen_kernels(13) gen_kernels(14) gen_kernels(15) gen_kernels(16)
    #undef gen_kernels
#endif

#undef gen_kernel


}
#endif
