

#include "../../../testing_commons/testing_flags.hpp"
#ifdef TEST_WARP_SHARED_VEC_REDUCTIONS
#include "../../../../../include/tk.metal"

namespace mittens {

#define gts_kern_PARAMS(T) \
    device T* _input [[buffer(0)]], \
    device T* _output [[buffer(1)]], \
    uint3         id              [[thread_position_in_grid]], \
    uint3         thread_group_id [[threadgroup_position_in_grid]], \
    uint          simd_group_id   [[simdgroup_index_in_threadgroup]], \
    uint          simd_lane_id    [[thread_index_in_simdgroup]]

template<typename T, unsigned L>
kernel void sv_reduce(gts_kern_PARAMS(T))
{
    using shared_vec  = sv<T, L* 8>;
    using GL = gl<T, 1, 1, 1, L * 8>;
    threadgroup shared_vec vec_a;
    GL input(_input, nullptr, nullptr, nullptr, nullptr);
    GL output(_output, nullptr, nullptr, nullptr, nullptr);
    load(vec_a, input, {0,0,0,0}, simd_lane_id);
    metal::simdgroup_barrier(metal::mem_flags::mem_none);
    T val = -2;
    metal::simdgroup_barrier(metal::mem_flags::mem_none);
    max(val, vec_a, simd_lane_id);
    metal::simdgroup_barrier(metal::mem_flags::mem_none);
    copy(vec_a, val, simd_lane_id);
    metal::simdgroup_barrier(metal::mem_flags::mem_none);
    store(output, vec_a, {0,0,0,0}, simd_lane_id);
}





#define gen_kernels(reg_N) \
    template [[host_name("t_sv_max_float_" #reg_N)]] [[kernel]] \
    void sv_reduce<float, reg_N>(gts_kern_PARAMS(float)); \
    template [[host_name("t_sv_max_bf16_" #reg_N)]] [[kernel]] \
    void sv_reduce<bf16, reg_N>(gts_kern_PARAMS(bf16)); \
    template [[host_name("t_sv_max_half_" #reg_N)]] [[kernel]] \
    void sv_reduce<half, reg_N>(gts_kern_PARAMS(half)); \



#if (INTENSITY_1)

    gen_kernels(1) gen_kernels(2)
    #undef gen_kernels
#elif (INTENSITY_2)
    gen_kernels(1) gen_kernels(2) gen_kernels(3) gen_kernels(4)
    #undef gen_kernels
#elif (INTENSITY_3) // 8

    gen_kernels(1) gen_kernels(2) gen_kernels(3) gen_kernels(4) gen_kernels(5) gen_kernels(6) gen_kernels(7) gen_kernels(8)
    #undef gen_kernels
#elif (INTENSITY_4) // 16

    gen_kernels(1) gen_kernels(2) gen_kernels(3) gen_kernels(4) gen_kernels(5) gen_kernels(6) gen_kernels(7) gen_kernels(8)
    gen_kernels(9) gen_kernels(10) gen_kernels(11) gen_kernels(12) gen_kernels(13) gen_kernels(14) gen_kernels(15) gen_kernels(16)
    #undef gen_kernels
#endif


}
#endif
