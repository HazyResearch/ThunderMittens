

#include "../../../testing_commons/testing_flags.hpp"
#ifdef TEST_GROUP_SHARED_TILE_CONVERSIONS
#include "../../../../../include/tk.metal"

namespace mittens {

#define gts_kern_PARAMS(T) \
    device T* _input [[buffer(0)]], \
    device T* _output [[buffer(1)]], \
    uint3         id              [[thread_position_in_grid]], \
    uint3         thread_group_id [[threadgroup_position_in_grid]], \
    uint          simd_group_id   [[simdgroup_index_in_threadgroup]], \
    uint          simd_lane_id    [[thread_index_in_simdgroup]] , \
    uint          threadIdx    [[thread_index_in_threadgroup]]

template<typename T, unsigned H, unsigned W, unsigned NW>
kernel void group_st_copy(gts_kern_PARAMS(T))
{
    using GL = gl<T, 1, 1, H * 8, W * 8>;
    using shared_tile = st<T, H * 8, W * 8>;
    using load_group = group<NW>;
    threadgroup shared_tile shared_A;
    threadgroup shared_tile shared_B;
    GL input(_input, nullptr, nullptr, nullptr, nullptr);
    GL output(_output, nullptr, nullptr, nullptr, nullptr);
    load_group::load(shared_A, input, {0,0,0,0}, threadIdx);
    metal::threadgroup_barrier(metal::mem_flags::mem_none);
    load_group::copy(shared_B, shared_A, threadIdx);
    metal::threadgroup_barrier(metal::mem_flags::mem_none);
    load_group::store(output, shared_B, {0,0,0,0}, threadIdx);
}


#define gen_kernel_NW(reg_N, reg_M, NW) \
    template [[host_name("group_st_copy_float_" #reg_N "x" #reg_M "_" #NW "warps")]] [[kernel]] \
    void group_st_copy<float, reg_N, reg_M, NW>(gts_kern_PARAMS(float)); \
    template [[host_name("group_st_copy_bf16_" #reg_N "x" #reg_M "_" #NW "warps")]] [[kernel]] \
    void group_st_copy<bf16, reg_N, reg_M, NW>(gts_kern_PARAMS(bf16)); \
    template [[host_name("group_st_copy_half_" #reg_N "x" #reg_M "_" #NW "warps")]] [[kernel]] \
    void group_st_copy<half, reg_N, reg_M, NW>(gts_kern_PARAMS(half)); \

#define gen_kernel(reg_N, reg_M) \
    gen_kernel_NW(reg_N, reg_M, 1) gen_kernel_NW(reg_N, reg_M, 2) \
    gen_kernel_NW(reg_N, reg_M, 3) gen_kernel_NW(reg_N, reg_M, 4)

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
