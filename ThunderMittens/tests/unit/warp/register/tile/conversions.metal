

#include "../../../testing_commons/testing_flags.hpp"
#ifdef TEST_WARP_REGISTER_TILE_CONVERSIONS
#include "../../../../../include/tk.metal"

namespace mittens {

#define gts_kern_PARAMS(T) \
    device T* _input [[buffer(0)]], \
    device T* _output [[buffer(1)]], \
    uint3         id              [[thread_position_in_grid]], \
    uint3         thread_group_id [[threadgroup_position_in_grid]], \
    uint          simd_group_id   [[simdgroup_index_in_threadgroup]], \
    uint          simd_lane_id    [[thread_index_in_simdgroup]]

//constant constexpr int B = 1, D = 1, R = 4, C = 5;
template<typename T, unsigned H, unsigned W, typename L>
kernel void rt_swap_layout(gts_kern_PARAMS(T))
{
    using reg_tile = rt<T, H * 8, W * 8, L>;
    using reg_tile_t = rt<T, H * 8, W * 8, typename ducks::rt_layout::transpose<L>::type>;
    using GL = gl<T, 1, 1, H * 8, W * 8>;
    reg_tile reg_A;
    reg_tile_t reg_B;
    GL input(_input, nullptr, nullptr, nullptr, nullptr);
    GL output(_output, nullptr, nullptr, nullptr, nullptr);
    load(reg_A, input, {0,0,0,0}, simd_lane_id);
    swap_layout(reg_B, reg_A, simd_lane_id);
    store(output, reg_B, {0,0,0,0}, simd_lane_id);
}
    
template<typename T, unsigned H, unsigned W, typename L>
kernel void rt_transpose(gts_kern_PARAMS(T))
{
    using reg_tile = rt<T, H * 8, W * 8, L>;
    using reg_tile_t = rt<T, W * 8, H * 8, L>;
    using GL = gl<T, 1, 1, H * 8, W * 8>;
    using GL_t = gl<T, 1, 1, W * 8, H * 8>;
    reg_tile reg_A;
    reg_tile_t reg_B;
    GL input(_input, nullptr, nullptr, nullptr, nullptr);
    GL_t output(_output, nullptr, nullptr, nullptr, nullptr);
    load(reg_A, input, {0,0,0,0}, simd_lane_id);
    transpose_sep(reg_B, reg_A, simd_lane_id);
    store(output, reg_B, {0,0,0,0}, simd_lane_id);
}
    
template<typename T, unsigned H, unsigned W, typename L>
kernel void rt_transpose_inplace(gts_kern_PARAMS(T))
{
    using reg_tile = rt<T, H * 8, H * 8, L>;
    using GL = gl<T, 1, 1, H * 8, H * 8>;
    reg_tile reg_A;
    GL input(_input, nullptr, nullptr, nullptr, nullptr);
    GL output(_output, nullptr, nullptr, nullptr, nullptr);
    load(reg_A, input, {0,0,0,0}, simd_lane_id);
    transpose_inplace(reg_A, simd_lane_id);
    store(output, reg_A, {0,0,0,0}, simd_lane_id);
}

template<typename T, unsigned H, unsigned W, typename L>
kernel void rt_make_causal(gts_kern_PARAMS(T))
{
    using reg_tile = rt<T, H * 8, H * 8, L>;
    using GL = gl<T, 1, 1, H * 8, H * 8>;
    reg_tile reg_A;
    GL input(_input, nullptr, nullptr, nullptr, nullptr);
    GL output(_output, nullptr, nullptr, nullptr, nullptr);
    load(reg_A, input, {0,0,0,0}, simd_lane_id);
    make_causal(reg_A, reg_A, simd_lane_id);
    store(output, reg_A, {0,0,0,0}, simd_lane_id);
}


#define gen_kernel(reg_N, reg_M) \
    template [[host_name("rt_swap_layout_float_" #reg_N "x" #reg_M "_rt_row_layout")]] [[kernel]] \
    void rt_swap_layout<float, reg_N, reg_M, ducks::rt_layout::row>(gts_kern_PARAMS(float)); \
    template [[host_name("rt_swap_layout_float_" #reg_N "x" #reg_M "_rt_col_layout")]] [[kernel]] \
    void rt_swap_layout<float, reg_N, reg_M, ducks::rt_layout::col>(gts_kern_PARAMS(float)); \
    template [[host_name("rt_swap_layout_bf16_" #reg_N "x" #reg_M "_rt_row_layout")]] [[kernel]] \
    void rt_swap_layout<bf16, reg_N, reg_M, ducks::rt_layout::row>(gts_kern_PARAMS(bf16)); \
    template [[host_name("rt_swap_layout_bf16_" #reg_N "x" #reg_M "_rt_col_layout")]] [[kernel]] \
    void rt_swap_layout<bf16, reg_N, reg_M, ducks::rt_layout::col>(gts_kern_PARAMS(bf16)); \
    template [[host_name("rt_swap_layout_half_" #reg_N "x" #reg_M "_rt_row_layout")]] [[kernel]] \
    void rt_swap_layout<half, reg_N, reg_M, ducks::rt_layout::row>(gts_kern_PARAMS(half)); \
    template [[host_name("rt_swap_layout_half_" #reg_N "x" #reg_M "_rt_col_layout")]] [[kernel]] \
    void rt_swap_layout<half, reg_N, reg_M, ducks::rt_layout::col>(gts_kern_PARAMS(half)); \
\
    template [[host_name("rt_transpose_float_" #reg_N "x" #reg_M "_rt_row_layout")]] [[kernel]] \
    void rt_transpose<float, reg_N, reg_M, ducks::rt_layout::row>(gts_kern_PARAMS(float)); \
    template [[host_name("rt_transpose_float_" #reg_N "x" #reg_M "_rt_col_layout")]] [[kernel]] \
    void rt_transpose<float, reg_N, reg_M, ducks::rt_layout::col>(gts_kern_PARAMS(float)); \
    template [[host_name("rt_transpose_bf16_" #reg_N "x" #reg_M "_rt_row_layout")]] [[kernel]] \
    void rt_transpose<bf16, reg_N, reg_M, ducks::rt_layout::row>(gts_kern_PARAMS(bf16)); \
    template [[host_name("rt_transpose_bf16_" #reg_N "x" #reg_M "_rt_col_layout")]] [[kernel]] \
    void rt_transpose<bf16, reg_N, reg_M, ducks::rt_layout::col>(gts_kern_PARAMS(bf16)); \
    template [[host_name("rt_transpose_half_" #reg_N "x" #reg_M "_rt_row_layout")]] [[kernel]] \
    void rt_transpose<half, reg_N, reg_M, ducks::rt_layout::row>(gts_kern_PARAMS(half)); \
    template [[host_name("rt_transpose_half_" #reg_N "x" #reg_M "_rt_col_layout")]] [[kernel]] \
    void rt_transpose<half, reg_N, reg_M, ducks::rt_layout::col>(gts_kern_PARAMS(half)); \
\
    template [[host_name("rt_transpose_inplace_float_" #reg_N "x" #reg_M "_rt_row_layout")]] [[kernel]] \
    void rt_transpose_inplace<float, reg_N, reg_M, ducks::rt_layout::row>(gts_kern_PARAMS(float)); \
    template [[host_name("rt_transpose_inplace_float_" #reg_N "x" #reg_M "_rt_col_layout")]] [[kernel]] \
    void rt_transpose_inplace<float, reg_N, reg_M, ducks::rt_layout::col>(gts_kern_PARAMS(float)); \
    template [[host_name("rt_transpose_inplace_bf16_" #reg_N "x" #reg_M "_rt_row_layout")]] [[kernel]] \
    void rt_transpose_inplace<bf16, reg_N, reg_M, ducks::rt_layout::row>(gts_kern_PARAMS(bf16)); \
    template [[host_name("rt_transpose_inplace_bf16_" #reg_N "x" #reg_M "_rt_col_layout")]] [[kernel]] \
    void rt_transpose_inplace<bf16, reg_N, reg_M, ducks::rt_layout::col>(gts_kern_PARAMS(bf16)); \
    template [[host_name("rt_transpose_inplace_half_" #reg_N "x" #reg_M "_rt_row_layout")]] [[kernel]] \
    void rt_transpose_inplace<half, reg_N, reg_M, ducks::rt_layout::row>(gts_kern_PARAMS(half)); \
    template [[host_name("rt_transpose_inplace_half_" #reg_N "x" #reg_M "_rt_col_layout")]] [[kernel]] \
    void rt_transpose_inplace<half, reg_N, reg_M, ducks::rt_layout::col>(gts_kern_PARAMS(half)); \
\
    template [[host_name("rt_make_causal_float_" #reg_N "x" #reg_M "_rt_row_layout")]] [[kernel]] \
    void rt_make_causal<float, reg_N, reg_M, ducks::rt_layout::row>(gts_kern_PARAMS(float)); \
    template [[host_name("rt_make_causal_float_" #reg_N "x" #reg_M "_rt_col_layout")]] [[kernel]] \
    void rt_make_causal<float, reg_N, reg_M, ducks::rt_layout::col>(gts_kern_PARAMS(float)); \
    template [[host_name("rt_make_causal_bf16_" #reg_N "x" #reg_M "_rt_row_layout")]] [[kernel]] \
    void rt_make_causal<bf16, reg_N, reg_M, ducks::rt_layout::row>(gts_kern_PARAMS(bf16)); \
    template [[host_name("rt_make_causal_bf16_" #reg_N "x" #reg_M "_rt_col_layout")]] [[kernel]] \
    void rt_make_causal<bf16, reg_N, reg_M, ducks::rt_layout::col>(gts_kern_PARAMS(bf16)); \
    template [[host_name("rt_make_causal_half_" #reg_N "x" #reg_M "_rt_row_layout")]] [[kernel]] \
    void rt_make_causal<half, reg_N, reg_M, ducks::rt_layout::row>(gts_kern_PARAMS(half)); \
    template [[host_name("rt_make_causal_half_" #reg_N "x" #reg_M "_rt_col_layout")]] [[kernel]] \
    void rt_make_causal<half, reg_N, reg_M, ducks::rt_layout::col>(gts_kern_PARAMS(half));


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
