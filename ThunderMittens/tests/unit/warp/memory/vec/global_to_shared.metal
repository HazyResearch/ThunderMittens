#include "../../../testing_commons/testing_flags.hpp"
#ifdef TEST_WARP_MEMORY_VEC_GLOBAL_TO_SHARED

#include "../../../../../include/tk.metal"
namespace mittens {

#define gtr_kern_PARAMS(T) \
    device T* _input [[buffer(0)]], \
    device T* _output [[buffer(1)]], \
    uint3         id              [[thread_position_in_grid]], \
    uint3         thread_group_id [[threadgroup_position_in_grid]], \
    uint          simd_group_id   [[simdgroup_index_in_threadgroup]], \
    uint          simd_lane_id    [[thread_index_in_simdgroup]]
constant constexpr int B = 1, D = 1, R = 1, C = 1;
template<typename T, unsigned L>
kernel void vec_gts(gtr_kern_PARAMS(T))
{
    using shared_vec = sv<T, L * 8>;
    using GL = gl<T, B, D, R, 8*L>;
    threadgroup shared_vec smemA;
    GL input(_input, nullptr, nullptr, nullptr, nullptr);
    GL output(_output, nullptr, nullptr, nullptr, nullptr);
    load(smemA, input, {0,0,0,0}, simd_lane_id);
    store(output, smemA, {0,0,0,0}, simd_lane_id);
}

#define gen_kernel(L) \
    template [[host_name("shared_vec_loadstore_gmem_float_" #L)]] [[kernel]] \
    void vec_gts<float, L>(gtr_kern_PARAMS(float)); \
    template [[host_name("shared_vec_loadstore_gmem_bf16_" #L)]] [[kernel]] \
    void vec_gts<bf16,  L>(gtr_kern_PARAMS(bf16)); \
    template [[host_name("shared_vec_loadstore_gmem_half_" #L)]] [[kernel]] \
    void vec_gts<half,  L>(gtr_kern_PARAMS(half));


#if (TEST_INTENSITY==1)
    gen_kernel(1) gen_kernel(2)
#elif (TEST_INTENSITY==2)
    gen_kernel(1) gen_kernel(2) gen_kernel(3) gen_kernel(4)
#elif (TEST_INTENSITY==3) // 8
    gen_kernel(1) gen_kernel(2) gen_kernel(3) gen_kernel(4) gen_kernel(5) gen_kernel(6) gen_kernel(7) gen_kernel(8)
#elif (TEST_INTENSITY==4) // 16
    gen_kernel(1) gen_kernel(2)  gen_kernel(3)  gen_kernel(4)  gen_kernel(5)  gen_kernel(6)  gen_kernel(7)  gen_kernel(8)
    gen_kernel(9) gen_kernel(10) gen_kernel(11) gen_kernel(12) gen_kernel(13) gen_kernel(14) gen_kernel(15) gen_kernel(16)
#endif

#undef gen_kernel
    

}

#endif
