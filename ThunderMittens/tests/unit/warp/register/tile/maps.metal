
#include "../../../testing_commons/testing_flags.hpp"
#ifdef TEST_WARP_REGISTER_TILE_MAPS
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
kernel void rt_unary(gts_kern_PARAMS(T))
{
    using reg_tile = rt<T, H * 8, W * 8, L>;
    using GL = gl<T, 1, 1, H * 8, W * 8>;
    reg_tile reg_A, reg_B;
    GL input(_input, nullptr, nullptr, nullptr, nullptr);
    GL output(_output, nullptr, nullptr, nullptr, nullptr);
    load(reg_A, input, {0,0,0,0}, simd_lane_id);
    exp(reg_B, reg_A);
    store(output, reg_B, {0,0,0,0}, simd_lane_id);
}
    
template<typename T, unsigned H, unsigned W, typename L>
kernel void rt_bin_RT_T(gts_kern_PARAMS(T))
{
    using reg_tile = rt<T, H * 8, W * 8, L>;
    using GL = gl<T, 1, 1, H * 8, W * 8>;
    reg_tile reg_A, reg_B;
    GL input(_input, nullptr, nullptr, nullptr, nullptr);
    GL output(_output, nullptr, nullptr, nullptr, nullptr);
    load(reg_A, input, {0,0,0,0}, simd_lane_id);
    T val = base_types::convertor<T, float>::convert(0.0);
    
    max(reg_B, reg_A, val);
    store(output, reg_B, {0,0,0,0}, simd_lane_id);
}
    
template<typename T, unsigned H, unsigned W, typename L>
kernel void rt_bin_RT_RT(gts_kern_PARAMS(T))
{
    using reg_tile = rt<T, H * 8, W * 8, L>;
    using GL = gl<T, 1, 1, H * 8, W * 8>;
    reg_tile reg_A, reg_B;
    GL input(_input, nullptr, nullptr, nullptr, nullptr);
    GL output(_output, nullptr, nullptr, nullptr, nullptr);
    load(reg_A, input, {0,0,0,0}, simd_lane_id);
    load(reg_B, input, {0,0,0,0}, simd_lane_id);
    
    add(reg_B, reg_A, reg_B);
    store(output, reg_B, {0,0,0,0}, simd_lane_id);
}

template<typename T, unsigned H, unsigned W, typename L>
kernel void rt_row_map(gts_kern_PARAMS(T))
{
    using reg_tile = rt<T, H * 8, W * 8, L>;
    using GL = gl<T, 1, 1, H * 8, W * 8>;
    using reg_vec = typename reg_tile::col_vec;
    
    reg_tile reg_A, reg_B;
    reg_vec reg_v;
    GL input(_input, nullptr, nullptr, nullptr, nullptr);
    GL output(_output, nullptr, nullptr, nullptr, nullptr);
    load(reg_A, input, {0,0,0,0}, simd_lane_id);
    load(reg_v, input, {0,0,0,0}, simd_lane_id);
    add_row(reg_B, reg_A, reg_v);
    store(output, reg_B, {0,0,0,0}, simd_lane_id);
}

template<typename T, unsigned H, unsigned W, typename L>
kernel void rt_col_map(gts_kern_PARAMS(T))
{
    using reg_tile = rt<T, H * 8, W * 8, L>;
    using GL = gl<T, 1, 1, H * 8, W * 8>;
    using reg_vec = typename reg_tile::row_vec;
    
    reg_tile reg_A, reg_B;
    reg_vec reg_v;
    GL input(_input, nullptr, nullptr, nullptr, nullptr);
    GL output(_output, nullptr, nullptr, nullptr, nullptr);
    load(reg_A, input, {0,0,0,0}, simd_lane_id);
    load(reg_v, input, {0,0,0,0}, simd_lane_id);
    add_col(reg_B, reg_A, reg_v);
    store(output, reg_B, {0,0,0,0}, simd_lane_id);
}

    //bin_map(thread RT &dst, thread const RT &lhs, thread const RT &rhs) {

#define gen_kernel(reg_N, reg_M) \
    template [[host_name("rt_exp_float_" #reg_N "x" #reg_M "_rt_row_layout")]] [[kernel]] \
    void rt_unary<float, reg_N, reg_M, ducks::rt_layout::row>(gts_kern_PARAMS(float)); \
    template [[host_name("rt_exp_float_" #reg_N "x" #reg_M "_rt_col_layout")]] [[kernel]] \
    void rt_unary<float, reg_N, reg_M, ducks::rt_layout::col>(gts_kern_PARAMS(float)); \
    template [[host_name("rt_exp_bf16_" #reg_N "x" #reg_M "_rt_row_layout")]] [[kernel]] \
    void rt_unary<bf16, reg_N, reg_M, ducks::rt_layout::row>(gts_kern_PARAMS(bf16)); \
    template [[host_name("rt_exp_bf16_" #reg_N "x" #reg_M "_rt_col_layout")]] [[kernel]] \
    void rt_unary<bf16, reg_N, reg_M, ducks::rt_layout::col>(gts_kern_PARAMS(bf16)); \
    template [[host_name("rt_exp_half_" #reg_N "x" #reg_M "_rt_row_layout")]] [[kernel]] \
    void rt_unary<half, reg_N, reg_M, ducks::rt_layout::row>(gts_kern_PARAMS(half)); \
    template [[host_name("rt_exp_half_" #reg_N "x" #reg_M "_rt_col_layout")]] [[kernel]] \
    void rt_unary<half, reg_N, reg_M, ducks::rt_layout::col>(gts_kern_PARAMS(half)); \
\
    template [[host_name("rt_t_max_float_" #reg_N "x" #reg_M "_rt_row_layout")]] [[kernel]] \
    void rt_bin_RT_T<float, reg_N, reg_M, ducks::rt_layout::row>(gts_kern_PARAMS(float)); \
    template [[host_name("rt_t_max_float_" #reg_N "x" #reg_M "_rt_col_layout")]] [[kernel]] \
    void rt_bin_RT_T<float, reg_N, reg_M, ducks::rt_layout::col>(gts_kern_PARAMS(float)); \
    template [[host_name("rt_t_max_bf16_" #reg_N "x" #reg_M "_rt_row_layout")]] [[kernel]] \
    void rt_bin_RT_T<bf16, reg_N, reg_M, ducks::rt_layout::row>(gts_kern_PARAMS(bf16)); \
    template [[host_name("rt_t_max_bf16_" #reg_N "x" #reg_M "_rt_col_layout")]] [[kernel]] \
    void rt_bin_RT_T<bf16, reg_N, reg_M, ducks::rt_layout::col>(gts_kern_PARAMS(bf16)); \
    template [[host_name("rt_t_max_half_" #reg_N "x" #reg_M "_rt_row_layout")]] [[kernel]] \
    void rt_bin_RT_T<half, reg_N, reg_M, ducks::rt_layout::row>(gts_kern_PARAMS(half)); \
    template [[host_name("rt_t_max_half_" #reg_N "x" #reg_M "_rt_col_layout")]] [[kernel]] \
    void rt_bin_RT_T<half, reg_N, reg_M, ducks::rt_layout::col>(gts_kern_PARAMS(half)); \
\
    template [[host_name("rt_rt_add_float_" #reg_N "x" #reg_M "_rt_row_layout")]] [[kernel]] \
    void rt_bin_RT_RT<float, reg_N, reg_M, ducks::rt_layout::row>(gts_kern_PARAMS(float)); \
    template [[host_name("rt_rt_add_float_" #reg_N "x" #reg_M "_rt_col_layout")]] [[kernel]] \
    void rt_bin_RT_RT<float, reg_N, reg_M, ducks::rt_layout::col>(gts_kern_PARAMS(float)); \
    template [[host_name("rt_rt_add_bf16_" #reg_N "x" #reg_M "_rt_row_layout")]] [[kernel]] \
    void rt_bin_RT_RT<bf16, reg_N, reg_M, ducks::rt_layout::row>(gts_kern_PARAMS(bf16)); \
    template [[host_name("rt_rt_add_bf16_" #reg_N "x" #reg_M "_rt_col_layout")]] [[kernel]] \
    void rt_bin_RT_RT<bf16, reg_N, reg_M, ducks::rt_layout::col>(gts_kern_PARAMS(bf16)); \
    template [[host_name("rt_rt_add_half_" #reg_N "x" #reg_M "_rt_row_layout")]] [[kernel]] \
    void rt_bin_RT_RT<half, reg_N, reg_M, ducks::rt_layout::row>(gts_kern_PARAMS(half)); \
    template [[host_name("rt_rt_add_half_" #reg_N "x" #reg_M "_rt_col_layout")]] [[kernel]] \
    void rt_bin_RT_RT<half, reg_N, reg_M, ducks::rt_layout::col>(gts_kern_PARAMS(half)); \
\
    template [[host_name("rt_v_row_add_float_" #reg_N "x" #reg_M "_rt_row_layout")]] [[kernel]] \
    void rt_row_map<float, reg_N, reg_M, ducks::rt_layout::row>(gts_kern_PARAMS(float)); \
    template [[host_name("rt_v_row_add_float_" #reg_N "x" #reg_M "_rt_col_layout")]] [[kernel]] \
    void rt_row_map<float, reg_N, reg_M, ducks::rt_layout::col>(gts_kern_PARAMS(float)); \
    template [[host_name("rt_v_row_add_bf16_" #reg_N "x" #reg_M "_rt_row_layout")]] [[kernel]] \
    void rt_row_map<bf16, reg_N, reg_M, ducks::rt_layout::row>(gts_kern_PARAMS(bf16)); \
    template [[host_name("rt_v_row_add_bf16_" #reg_N "x" #reg_M "_rt_col_layout")]] [[kernel]] \
    void rt_row_map<bf16, reg_N, reg_M, ducks::rt_layout::col>(gts_kern_PARAMS(bf16)); \
    template [[host_name("rt_v_row_add_half_" #reg_N "x" #reg_M "_rt_row_layout")]] [[kernel]] \
    void rt_row_map<half, reg_N, reg_M, ducks::rt_layout::row>(gts_kern_PARAMS(half)); \
    template [[host_name("rt_v_row_add_half_" #reg_N "x" #reg_M "_rt_col_layout")]] [[kernel]] \
    void rt_row_map<half, reg_N, reg_M, ducks::rt_layout::col>(gts_kern_PARAMS(half)); \
\
    template [[host_name("rt_v_col_add_float_" #reg_N "x" #reg_M "_rt_row_layout")]] [[kernel]] \
    void rt_col_map<float, reg_N, reg_M, ducks::rt_layout::row>(gts_kern_PARAMS(float)); \
    template [[host_name("rt_v_col_add_float_" #reg_N "x" #reg_M "_rt_col_layout")]] [[kernel]] \
    void rt_col_map<float, reg_N, reg_M, ducks::rt_layout::col>(gts_kern_PARAMS(float)); \
    template [[host_name("rt_v_col_add_bf16_" #reg_N "x" #reg_M "_rt_row_layout")]] [[kernel]] \
    void rt_col_map<bf16, reg_N, reg_M, ducks::rt_layout::row>(gts_kern_PARAMS(bf16)); \
    template [[host_name("rt_v_col_add_bf16_" #reg_N "x" #reg_M "_rt_col_layout")]] [[kernel]] \
    void rt_col_map<bf16, reg_N, reg_M, ducks::rt_layout::col>(gts_kern_PARAMS(bf16)); \
    template [[host_name("rt_v_col_add_half_" #reg_N "x" #reg_M "_rt_row_layout")]] [[kernel]] \
    void rt_col_map<half, reg_N, reg_M, ducks::rt_layout::row>(gts_kern_PARAMS(half)); \
    template [[host_name("rt_v_col_add_half_" #reg_N "x" #reg_M "_rt_col_layout")]] [[kernel]] \
    void rt_col_map<half, reg_N, reg_M, ducks::rt_layout::col>(gts_kern_PARAMS(half));


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
