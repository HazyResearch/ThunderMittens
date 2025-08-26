#include "../../../testing_commons/testing_flags.hpp"
#ifdef TEST_WARP_REGISTER_VEC_CONVERSIONS
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
template<typename T, unsigned W, typename L1, typename L2>
kernel void rv_copy(gts_kern_PARAMS(T))
{
    using reg_vec1 = rv<T, W * 8, L1>;
    using reg_vec2 = rv<T, W * 8, L2>;
    using GL = gl<T, 1, 1, 1, W * 8>;
    reg_vec1 vec_a;
    reg_vec2 vec_b;
    GL input(_input, nullptr, nullptr, nullptr, nullptr);
    GL output(_output, nullptr, nullptr, nullptr, nullptr);
    load(vec_a, input, {0,0,0,0}, simd_lane_id);
    copy(vec_b, vec_a, simd_lane_id);
    store(output, vec_b, {0,0,0,0}, simd_lane_id);
}
    
#define gen_kernel_LL(reg_N, L1, L1_name, L2, L2_name) \
    template [[host_name("rv_copy_float_" #reg_N "_" #L1_name "_" #L2_name)]] [[kernel]] \
    void rv_copy<float, reg_N, L1, L2>(gts_kern_PARAMS(float));                          \
    template [[host_name("rv_copy_half_" #reg_N "_" #L1_name "_" #L2_name)]] [[kernel]]  \
    void rv_copy<half, reg_N, L1, L2>(gts_kern_PARAMS(half));                            \
    template [[host_name("rv_copy_bf16_" #reg_N "_" #L1_name "_" #L2_name)]] [[kernel]]  \
    void rv_copy<bf16, reg_N, L1, L2>(gts_kern_PARAMS(bf16));
    
#define gen_kernels(L) \
    gen_kernel_LL(L, ducks::rv_layout::align, rv_align_layout, ducks::rv_layout::align, rv_align_layout) \
    gen_kernel_LL(L, ducks::rv_layout::align, rv_align_layout, ducks::rv_layout::ortho, rv_ortho_layout) \
    gen_kernel_LL(L, ducks::rv_layout::align, rv_align_layout, ducks::rv_layout::naive, rv_naive_layout) \
\
    gen_kernel_LL(L, ducks::rv_layout::ortho, rv_ortho_layout, ducks::rv_layout::align, rv_align_layout) \
    gen_kernel_LL(L, ducks::rv_layout::ortho, rv_ortho_layout, ducks::rv_layout::ortho, rv_ortho_layout) \
    gen_kernel_LL(L, ducks::rv_layout::ortho, rv_ortho_layout, ducks::rv_layout::naive, rv_naive_layout) \
\
    gen_kernel_LL(L, ducks::rv_layout::naive, rv_naive_layout, ducks::rv_layout::align, rv_align_layout) \
    gen_kernel_LL(L, ducks::rv_layout::naive, rv_naive_layout, ducks::rv_layout::ortho, rv_ortho_layout) \
    gen_kernel_LL(L, ducks::rv_layout::naive, rv_naive_layout, ducks::rv_layout::naive, rv_naive_layout) 

    
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

#undef gen_kernel


}
#endif
