
#include "../../../testing_commons/testing_flags.hpp"
#ifdef TEST_WARP_REGISTER_TILE_MMA
#include "../../../../../include/tk.metal"

namespace mittens {

#define gts_kern_PARAMS(T) \
    device T* _input [[buffer(0)]], \
    device T* _output [[buffer(1)]], \
    uint3         id              [[thread_position_in_grid]], \
    uint3         thread_group_id [[threadgroup_position_in_grid]], \
    uint          simd_group_id   [[simdgroup_index_in_threadgroup]], \
    uint          simd_lane_id    [[thread_index_in_simdgroup]]

template<unsigned N, unsigned K, unsigned M>
kernel void rt_mma_AB(gts_kern_PARAMS(float))
{
    rt<float, N * 8, K * 8> reg_A;
    rt<float, K * 8, M * 8> reg_B;
    rt<float, N * 8, M * 8> reg_C;
    using GL_A = gl<float, 1, 1, N * 8, K * 8>;
    using GL_B = gl<float, 1, 1, K * 8, M * 8>;
    using GL_C = gl<float, 1, 1, N * 8, M * 8>;
    GL_A input_A(_input, nullptr, nullptr, nullptr, nullptr);
    GL_B input_B(_input + N * K * 64, nullptr, nullptr, nullptr, nullptr);
    GL_C output(_output, nullptr, nullptr, nullptr, nullptr);
    load(reg_A, input_A, {0,0,0,0}, simd_lane_id);
    load(reg_B, input_B, {0,0,0,0}, simd_lane_id);
    zero(reg_C);
    metal::simdgroup_barrier(metal::mem_flags::mem_none);
    mma_AB(reg_C, reg_A, reg_B, reg_C);
    metal::simdgroup_barrier(metal::mem_flags::mem_none);
    store(output, reg_C, {0,0,0,0}, simd_lane_id);
}
    
template<unsigned N, unsigned K, unsigned M>
kernel void rt_mma_ABt(gts_kern_PARAMS(float))
{
    rt<float, N * 8, K * 8, ducks::rt_layout::row> reg_A;
    rt<float, M * 8, K * 8, ducks::rt_layout::col> reg_B;
    rt<float, N * 8, M * 8, ducks::rt_layout::row> reg_C;
    
    gl<float, 1, 1, N * 8, K * 8> input_A(_input, nullptr, nullptr, nullptr, nullptr);
    gl<float, 1, 1, M * 8, K * 8> input_B(_input + N * K * 64, nullptr, nullptr, nullptr, nullptr);
    gl<float, 1, 1, N * 8, M * 8> output(_output, nullptr, nullptr, nullptr, nullptr);
    
    load(reg_A, input_A, {0,0,0,0}, simd_lane_id);
    load(reg_B, input_B, {0,0,0,0}, simd_lane_id);
    zero(reg_C);
    metal::simdgroup_barrier(metal::mem_flags::mem_device);
    mma_ABt(reg_C, reg_A, reg_B, reg_C);
    metal::simdgroup_barrier(metal::mem_flags::mem_device);
    store(output, reg_C, {0,0,0,0}, simd_lane_id);
    metal::simdgroup_barrier(metal::mem_flags::mem_device);
}

template<unsigned N, unsigned K, unsigned M>
kernel void rt_mma_AtB(gts_kern_PARAMS(float))
{
    rt<float, K * 8, N * 8, ducks::rt_layout::col> reg_A;
    rt<float, K * 8, M * 8> reg_B;
    rt<float, N * 8, M * 8> reg_C;
    using GL_A = gl<float, 1, 1, K * 8, N * 8>;
    using GL_B = gl<float, 1, 1, K * 8, M * 8>;
    using GL_C = gl<float, 1, 1, N * 8, M * 8>;
    GL_A input_A(_input, nullptr, nullptr, nullptr, nullptr);
    GL_B input_B(_input + N * K * 64, nullptr, nullptr, nullptr, nullptr);
    GL_C output(_output, nullptr, nullptr, nullptr, nullptr);
    load(reg_A, input_A, {0,0,0,0}, simd_lane_id);
    load(reg_B, input_B, {0,0,0,0}, simd_lane_id);
    zero(reg_C);
    metal::simdgroup_barrier(metal::mem_flags::mem_none);
    mma_AtB(reg_C, reg_A, reg_B, reg_C);
    metal::simdgroup_barrier(metal::mem_flags::mem_none);
    store(output, reg_C, {0,0,0,0}, simd_lane_id);
}

template<unsigned N, unsigned K, unsigned M>
kernel void rt_mma_AtBt(gts_kern_PARAMS(float))
{
    rt<float, K * 8, N * 8, ducks::rt_layout::col> reg_A;
    rt<float, M * 8, K * 8, ducks::rt_layout::col> reg_B;
    rt<float, N * 8, M * 8> reg_C;
    using GL_A = gl<float, 1, 1, K * 8, N * 8>;
    using GL_B = gl<float, 1, 1, M * 8, K * 8>;
    using GL_C = gl<float, 1, 1, N * 8, M * 8>;
    GL_A input_A(_input, nullptr, nullptr, nullptr, nullptr);
    GL_B input_B(_input + N * K * 64, nullptr, nullptr, nullptr, nullptr);
    GL_C output(_output, nullptr, nullptr, nullptr, nullptr);
    load(reg_A, input_A, {0,0,0,0}, simd_lane_id);
    load(reg_B, input_B, {0,0,0,0}, simd_lane_id);
    zero(reg_C);
    metal::simdgroup_barrier(metal::mem_flags::mem_device);
    mma_AtBt(reg_C, reg_A, reg_B, reg_C);
    metal::simdgroup_barrier(metal::mem_flags::mem_device);
    store(output, reg_C, {0,0,0,0}, simd_lane_id);
}
    
    
#define gen_kernel(N, K, M) \
    template [[host_name("reg_mma_AB_" #N "x" #K "x" #M)]] [[kernel]] \
    void rt_mma_AB<N, K, M>(gts_kern_PARAMS(float)); \
    template [[host_name("reg_mma_ABt_" #N "x" #K "x" #M)]] [[kernel]] \
    void rt_mma_ABt<N, K, M>(gts_kern_PARAMS(float)); \
    template [[host_name("reg_mma_AtB_" #N "x" #K "x" #M)]] [[kernel]] \
    void rt_mma_AtB<N, K, M>(gts_kern_PARAMS(float)); \
    template [[host_name("reg_mma_AtBt_" #N "x" #K "x" #M)]] [[kernel]] \
    void rt_mma_AtBt<N, K, M>(gts_kern_PARAMS(float));

#if (INTENSITY_1)
    #define gen_kernels(N) \
        gen_kernel(N, 1, 1) gen_kernel(N, 1, 2) \
        gen_kernel(N, 2, 1) gen_kernel(N, 2, 2) \
        gen_kernel(N, 3, 1) gen_kernel(N, 3, 2) \
        gen_kernel(N, 4, 1) gen_kernel(N, 4, 2)

    gen_kernels(1) gen_kernels(2)
    #undef gen_kernels

#elif (INTENSITY_2)
    #define gen_kernels(N) \
        gen_kernel(N, 1, 1) gen_kernel(N, 1, 2) gen_kernel(N, 1, 3) gen_kernel(N, 1, 4) \
        gen_kernel(N, 2, 1) gen_kernel(N, 2, 2) gen_kernel(N, 2, 3) gen_kernel(N, 2, 4) \
        gen_kernel(N, 3, 1) gen_kernel(N, 3, 2) gen_kernel(N, 3, 3) gen_kernel(N, 3, 4) \
        gen_kernel(N, 4, 1) gen_kernel(N, 4, 2) gen_kernel(N, 4, 3) gen_kernel(N, 4, 4)

    gen_kernels(1) gen_kernels(2) gen_kernels(3) gen_kernels(4)
    #undef gen_kernels
#elif (INTENSITY_3) // 8
    
    #define gen_kernels(N) \
        gen_kernel(N, 1, 1) gen_kernel(N, 1, 2) gen_kernel(N, 1, 3) gen_kernel(N, 1, 4) gen_kernel(N, 1, 5) gen_kernel(N, 1, 6) gen_kernel(N, 1, 7) gen_kernel(N, 1, 8) \
        gen_kernel(N, 2, 1) gen_kernel(N, 2, 2) gen_kernel(N, 2, 3) gen_kernel(N, 2, 4) gen_kernel(N, 2, 5) gen_kernel(N, 2, 6) gen_kernel(N, 2, 7) gen_kernel(N, 2, 8) \
        gen_kernel(N, 3, 1) gen_kernel(N, 3, 2) gen_kernel(N, 3, 3) gen_kernel(N, 3, 4) gen_kernel(N, 3, 5) gen_kernel(N, 3, 6) gen_kernel(N, 3, 7) gen_kernel(N, 3, 8) \
        gen_kernel(N, 4, 1) gen_kernel(N, 4, 2) gen_kernel(N, 4, 3) gen_kernel(N, 4, 4) gen_kernel(N, 4, 5) gen_kernel(N, 4, 6) gen_kernel(N, 4, 7) gen_kernel(N, 4, 8)

    gen_kernels(1) gen_kernels(2) gen_kernels(3) gen_kernels(4) gen_kernels(5) gen_kernels(6) gen_kernels(7) gen_kernels(8)
    #undef gen_kernels
#elif (INTENSITY_4) // 16
    #define gen_kernels(N) \
        gen_kernel(N, 1, 1) gen_kernel(N, 1, 2)  gen_kernel(N, 1, 3)  gen_kernel(N, 1, 4)  gen_kernel(N, 1, 5)  gen_kernel(N, 1, 6)  gen_kernel(N, 1, 7)  gen_kernel(N, 1, 8)  \
        gen_kernel(N, 1, 9) gen_kernel(N, 1, 10) gen_kernel(N, 1, 11) gen_kernel(N, 1, 12) gen_kernel(N, 1, 13) gen_kernel(N, 1, 14) gen_kernel(N, 1, 15) gen_kernel(N, 1, 16) \
        gen_kernel(N, 2, 1) gen_kernel(N, 2, 2)  gen_kernel(N, 2, 3)  gen_kernel(N, 2, 4)  gen_kernel(N, 2, 5)  gen_kernel(N, 2, 6)  gen_kernel(N, 2, 7)  gen_kernel(N, 2, 8)  \
        gen_kernel(N, 2, 9) gen_kernel(N, 2, 10) gen_kernel(N, 2, 11) gen_kernel(N, 2, 12) gen_kernel(N, 2, 13) gen_kernel(N, 2, 14) gen_kernel(N, 2, 15) gen_kernel(N, 2, 16) \
        gen_kernel(N, 3, 1) gen_kernel(N, 3, 2)  gen_kernel(N, 3, 3)  gen_kernel(N, 3, 4)  gen_kernel(N, 3, 5)  gen_kernel(N, 3, 6)  gen_kernel(N, 3, 7)  gen_kernel(N, 3, 8)  \
        gen_kernel(N, 3, 9) gen_kernel(N, 3, 10) gen_kernel(N, 3, 11) gen_kernel(N, 3, 12) gen_kernel(N, 3, 13) gen_kernel(N, 3, 14) gen_kernel(N, 3, 15) gen_kernel(N, 3, 16) \
        gen_kernel(N, 4, 1) gen_kernel(N, 4, 2)  gen_kernel(N, 4, 3)  gen_kernel(N, 4, 4)  gen_kernel(N, 4, 5)  gen_kernel(N, 4, 6)  gen_kernel(N, 4, 7)  gen_kernel(N, 4, 8)  \
        gen_kernel(N, 4, 9) gen_kernel(N, 4, 10) gen_kernel(N, 4, 11) gen_kernel(N, 4, 12) gen_kernel(N, 4, 13) gen_kernel(N, 4, 14) gen_kernel(N, 4, 15) gen_kernel(N, 4, 16)

    gen_kernels(1)
    gen_kernels(2)
    gen_kernels(3) gen_kernels(4) gen_kernels(5) gen_kernels(6) gen_kernels(7) gen_kernels(8)
    gen_kernels(9) gen_kernels(10) gen_kernels(11) gen_kernels(12) gen_kernels(13) gen_kernels(14) gen_kernels(15) gen_kernels(16)
    
    #undef gen_kernels
#endif

#undef gen_kernel

}

#endif
