

#include "../../../testing_commons/testing_flags.hpp"
#ifdef TEST_GROUP_SHARED_VEC_MAPS
#include "../../../../../include/tk.metal"

namespace mittens {

#define gts_kern_PARAMS(T) \
    device T* _input [[buffer(0)]], \
    device T* _output [[buffer(1)]], \
    uint3         id              [[thread_position_in_grid]], \
    uint3         thread_group_id [[threadgroup_position_in_grid]], \
    uint          simd_group_id   [[simdgroup_index_in_threadgroup]], \
    uint          simd_lane_id    [[thread_index_in_simdgroup]], \
    uint          threadIdx       [[thread_index_in_threadgroup]] \

template<typename T, unsigned L, unsigned NW>
kernel void group_sv_unary(gts_kern_PARAMS(T))
{
    using load_group = group<NW>;
    using shared_vec  = sv<T, L* 8>;
    using GL = gl<T, 1, 1, 1, L * 8>;
    threadgroup shared_vec vec_a;
    threadgroup shared_vec vec_b;
    GL input(_input, nullptr, nullptr, nullptr, nullptr);
    GL output(_output, nullptr, nullptr, nullptr, nullptr);
    load_group::load(vec_a, input, {0,0,0,0}, threadIdx);
    metal::threadgroup_barrier(metal::mem_flags::mem_none);
    load_group::exp(vec_b, vec_a, threadIdx);
    metal::threadgroup_barrier(metal::mem_flags::mem_none);
    load_group::store(output, vec_b, {0,0,0,0}, threadIdx);
}

template<typename T, unsigned L, unsigned NW>
kernel void group_sv_sv_binary(gts_kern_PARAMS(T))
{
    using load_group = group<NW>;
    using shared_vec  = sv<T, L* 8>;
    using GL = gl<T, 1, 1, 1, L * 8>;
    threadgroup shared_vec vec_a;
    threadgroup shared_vec vec_b;
    GL input(_input, nullptr, nullptr, nullptr, nullptr);
    GL output(_output, nullptr, nullptr, nullptr, nullptr);
    load_group::load(vec_a, input, {0,0,0,0}, threadIdx);
    load_group::load(vec_b, input, {0,0,0,0}, threadIdx);
    metal::threadgroup_barrier(metal::mem_flags::mem_none);
    load_group::add(vec_b, vec_a, vec_b, threadIdx);
    metal::threadgroup_barrier(metal::mem_flags::mem_none);
    load_group::store(output, vec_b, {0,0,0,0}, threadIdx);
}
    
template<typename T, unsigned L, unsigned NW>
kernel void group_sv_t_binary(gts_kern_PARAMS(T))
{
    using load_group = group<NW>;
    using shared_vec  = sv<T, L* 8>;
    using GL = gl<T, 1, 1, 1, L * 8>;
    threadgroup shared_vec vec_a;
    threadgroup shared_vec vec_b;
    GL input(_input, nullptr, nullptr, nullptr, nullptr);
    GL output(_output, nullptr, nullptr, nullptr, nullptr);
    load_group::load(vec_a, input, {0,0,0,0}, threadIdx);
    T val = base_types::convertor<T, float>::convert(0.5f);
    metal::threadgroup_barrier(metal::mem_flags::mem_none);
    load_group::add(vec_b, vec_a, val, threadIdx);
    metal::threadgroup_barrier(metal::mem_flags::mem_none);
    load_group::store(output, vec_b, {0,0,0,0}, threadIdx);
}
    // group_sv_t_add



#define gen_kernels_NW(reg_N, NW) \
    template [[host_name("group_sv_exp_float_" #reg_N "_" #NW "warps")]] [[kernel]] \
    void group_sv_unary<float, reg_N, NW>(gts_kern_PARAMS(float)); \
    template [[host_name("group_sv_exp_bf16_"  #reg_N "_" #NW "warps")]] [[kernel]] \
    void group_sv_unary<bf16, reg_N, NW>(gts_kern_PARAMS(bf16)); \
    template [[host_name("group_sv_exp_half_"  #reg_N "_" #NW "warps")]] [[kernel]] \
    void group_sv_unary<half, reg_N, NW>(gts_kern_PARAMS(half)); \
\
    template [[host_name("group_sv_sv_add_float_" #reg_N "_" #NW "warps")]] [[kernel]] \
    void group_sv_sv_binary<float, reg_N, NW>(gts_kern_PARAMS(float)); \
    template [[host_name("group_sv_sv_add_bf16_"  #reg_N "_" #NW "warps")]] [[kernel]] \
    void group_sv_sv_binary<bf16, reg_N, NW>(gts_kern_PARAMS(bf16)); \
    template [[host_name("group_sv_sv_add_half_"  #reg_N "_" #NW "warps")]] [[kernel]] \
    void group_sv_sv_binary<half, reg_N, NW>(gts_kern_PARAMS(half)); \
\
    template [[host_name("group_sv_t_add_float_" #reg_N "_" #NW "warps")]] [[kernel]] \
    void group_sv_t_binary<float, reg_N, NW>(gts_kern_PARAMS(float)); \
    template [[host_name("group_sv_t_add_bf16_"  #reg_N "_" #NW "warps")]] [[kernel]] \
    void group_sv_t_binary<bf16, reg_N, NW>(gts_kern_PARAMS(bf16)); \
    template [[host_name("group_sv_t_add_half_"  #reg_N "_" #NW "warps")]] [[kernel]] \
    void group_sv_t_binary<half, reg_N, NW>(gts_kern_PARAMS(half)); \

    
    //group_sv_sv_binary
#define gen_kernels(reg_N) \
    gen_kernels_NW(reg_N, 2) gen_kernels_NW(reg_N, 3) gen_kernels_NW(reg_N, 4)
    
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
