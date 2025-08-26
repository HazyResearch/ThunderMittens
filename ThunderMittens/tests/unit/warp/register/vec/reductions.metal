#include "../../../testing_commons/testing_flags.hpp"
#ifdef TEST_WARP_REGISTER_VEC_REDUCTIONS
#include "../../../../../include/tk.metal"

namespace mittens {

#define gts_kern_PARAMS(T) \
    device T* _input [[buffer(0)]], \
    device T* _output [[buffer(1)]], \
    uint3         id              [[thread_position_in_grid]], \
    uint3         thread_group_id [[threadgroup_position_in_grid]], \
    uint          simd_group_id   [[simdgroup_index_in_threadgroup]], \
    uint          simd_lane_id    [[thread_index_in_simdgroup]]

template<typename T, unsigned W, typename L>
kernel void rv_reduce(gts_kern_PARAMS(T))
{
    using reg_vec1 = rv<T, W * 8, L>;
    using reg_vec2 = rv<T, W * 8, L>;
    using GL = gl<T, 1, 1, 1, W * 8>;
    reg_vec1 vec_a;
    reg_vec2 vec_b;
    GL input(_input, nullptr, nullptr, nullptr, nullptr);
    GL output(_output, nullptr, nullptr, nullptr, nullptr);
    load(vec_a, input, {0,0,0,0}, simd_lane_id);
    T dst;
    max(dst, vec_a, simd_lane_id);
    copy(vec_b, dst);
    store(output, vec_b, {0,0,0,0}, simd_lane_id);
}
    

#define gen_kernels(reg_N) \
    template [[host_name("rv_add_float_" #reg_N "_rv_align_layout")]] [[kernel]]  \
    void rv_reduce<float, reg_N, ducks::rv_layout::align>(gts_kern_PARAMS(float)); \
    template [[host_name("rv_add_float_" #reg_N "_rv_ortho_layout")]] [[kernel]]  \
    void rv_reduce<float, reg_N, ducks::rv_layout::ortho>(gts_kern_PARAMS(float)); \
    template [[host_name("rv_add_float_" #reg_N "_rv_naive_layout")]] [[kernel]]  \
    void rv_reduce<float, reg_N, ducks::rv_layout::naive>(gts_kern_PARAMS(float)); \
\
    template [[host_name("rv_add_half_" #reg_N "_rv_align_layout")]] [[kernel]]  \
    void rv_reduce<half, reg_N, ducks::rv_layout::align>(gts_kern_PARAMS(half));  \
    template [[host_name("rv_add_half_" #reg_N "_rv_ortho_layout")]] [[kernel]]  \
    void rv_reduce<half, reg_N, ducks::rv_layout::ortho>(gts_kern_PARAMS(half));  \
    template [[host_name("rv_add_half_" #reg_N "_rv_naive_layout")]] [[kernel]]  \
    void rv_reduce<half, reg_N, ducks::rv_layout::naive>(gts_kern_PARAMS(half));  \
\
    template [[host_name("rv_add_bf16_" #reg_N "_rv_align_layout")]] [[kernel]]  \
    void rv_reduce<bf16, reg_N, ducks::rv_layout::align>(gts_kern_PARAMS(bf16));  \
    template [[host_name("rv_add_bf16_" #reg_N "_rv_ortho_layout")]] [[kernel]]  \
    void rv_reduce<bf16, reg_N, ducks::rv_layout::ortho>(gts_kern_PARAMS(bf16));  \
    template [[host_name("rv_add_bf16_" #reg_N "_rv_naive_layout")]] [[kernel]]  \
    void rv_reduce<bf16, reg_N, ducks::rv_layout::naive>(gts_kern_PARAMS(bf16));  \

    

    
#if (INTENSITY_1)
    gen_kernels(1) gen_kernels(2)
#elif (INTENSITY_2)
    gen_kernels(1) gen_kernels(2) gen_kernels(3) gen_kernels(4)
#elif (INTENSITY_3) // 8
    gen_kernels(1) gen_kernels(2) gen_kernels(3) gen_kernels(4) gen_kernels(5) gen_kernels(6) gen_kernels(7) gen_kernels(8)
#elif (INTENSITY_4) // 16
    gen_kernels(1) gen_kernels(2) gen_kernels(3) gen_kernels(4) gen_kernels(5) gen_kernels(6) gen_kernels(7) gen_kernels(8)
    gen_kernels(9) gen_kernels(10) gen_kernels(11) gen_kernels(12) gen_kernels(13) gen_kernels(14) gen_kernels(15) gen_kernels(16)
#endif

#undef gen_kernels

}

#endif


