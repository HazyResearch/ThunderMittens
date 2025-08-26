#include "tk.metal"
#include <metal_stdlib>
namespace mittens {

#define PARAMS(T) \
    device T* q [[buffer(0)]], \
    device T* k [[buffer(1)]], \
    device T* v [[buffer(2)]], \
    device T* o [[buffer(3)]], \
    constant unsigned &H [[buffer(4)]], \
    constant unsigned &N [[buffer(5)]], \
    uint3 blockIdx [[threadgroup_position_in_grid]], \
    uint  laneId   [[thread_index_in_simdgroup]] \

namespace custom_ops {
struct subexp2 {;
    template<typename T> static METAL_FUNC T op(thread const T &a, thread const T &b) { return metal::exp2(a-b); }
};
}
    
template<typename RT, typename RV>
static METAL_FUNC typename metal::enable_if<ducks::is_register_tile<RT>() && ducks::is_register_vector<RV>(), void>::type
subexp2(thread RT &dst, thread const RT &src, thread const RV &row_values) {
    row_map<custom_ops::subexp2, RT, RV>(dst, src, row_values);
}
template<typename RV, typename U>
static METAL_FUNC typename metal::enable_if<ducks::is_register_vector<RV>(), void>::type
subexp2(thread RV &dst, thread const RV &lhs, thread const U &rhs) {
    bin_op<custom_ops::subexp2, RV>(dst, lhs, rhs);
}

constant constexpr const int TN = 8;
template <int D>
kernel void attn_fwd(device   bf16     *q [[buffer(0)]],
                     device   bf16     *k [[buffer(1)]],
                     device   bf16     *v [[buffer(2)]],
                     device   bf16     *o [[buffer(3)]],
                     constant unsigned &N [[buffer(4)]],
                     constant unsigned &H [[buffer(5)]],
                     uint3 blockIdx [[threadgroup_position_in_grid]],
                     uint laneId [[thread_index_in_simdgroup]]) {
    static_assert(D == 64 || D == 128, "D must be 64 or 128");
    using global_layout = gl<bfloat, 1, -1, -1, D>;
    global_layout gl_q(q, nullptr, H, N, nullptr);
    global_layout gl_k(k, nullptr, H, N, nullptr);
    global_layout gl_v(v, nullptr, H, N, nullptr);
    global_layout gl_o(o, nullptr, H, N, nullptr);
    using st_qkv     = st_bf<TN, D>;
    using rt_qkv     = rt_bf<TN, D>;
    using rt_k_t     = rt_bf<TN, D, ducks::rt_layout::col>;
    using rt_att     = rt_fl<TN, TN>;
    using rt_att_mma = rt_bf<TN, TN>;
    using rt_o       = rt_fl<TN, D>;
    using rv_att     = rt_fl<TN, TN>::col_vec;

    const int block = blockIdx.z;
    const int head = blockIdx.y;
    const int q_seq = blockIdx.x;
    
    const int kv_blocks = N / st_qkv::rows;
    rt_qkv q_reg;
    rt_k_t k_reg;
    rt_qkv v_reg;
    rt_att att_block;
    rt_o o_reg;
    rv_att max_vec_last;
    rv_att max_vec;
    rv_att norm_vec;

    load(q_reg, gl_q, {block, head, q_seq, 0}, laneId);
    neg_infty(max_vec);
    zero(norm_vec);
    zero(o_reg);
    constexpr const bf16 q_mul = ((D == 128) ? 0.08838834764bf : 0.125bf) * 1.44269504089bf;
    mul(q_reg, q_reg, q_mul);
    #pragma clang loop unroll(full)
    for(auto kv_idx = 0; kv_idx < kv_blocks; kv_idx++) {
        load(k_reg, gl_k, {block, head, kv_idx, 0}, laneId);
        zero(att_block);
        mma_ABt(att_block, q_reg, k_reg, att_block);
        copy(max_vec_last,  max_vec, laneId);
        row_max(max_vec, att_block, max_vec, laneId);
        
//        subexp2(max_vec_last, max_vec_last, max_vec);
        sub(max_vec_last, max_vec_last, max_vec);
        exp2(max_vec_last, max_vec_last);
//        subexp2(att_block, att_block, max_vec);
        
        sub_row(att_block, att_block, max_vec);
        exp2(att_block, att_block);
        
        mul(norm_vec, norm_vec, max_vec_last);
        row_sum(norm_vec, att_block, norm_vec, laneId);
        mul_row(o_reg, o_reg, max_vec_last);
        load(v_reg, gl_v, {block, head, kv_idx, 0}, laneId);
        mma_AB(o_reg, att_block, v_reg, o_reg);
    }
    div_row(o_reg, o_reg, norm_vec);
    store(gl_o, o_reg, {block, head, q_seq, 0}, laneId);
}


#define instantiate_add_custom(D)                                \
  template [[host_name("attn_fwd_" #D)]] [[kernel]] void         \
  attn_fwd<D>(device   bf16     *q [[buffer(0)]], \
    device   bf16     *k [[buffer(1)]], \
    device   bf16     *v [[buffer(2)]], \
    device   bf16     *o [[buffer(3)]], \
    constant unsigned &N [[buffer(4)]], \
    constant unsigned &H [[buffer(5)]], \
    uint3 blockIdx [[threadgroup_position_in_grid]], \
    uint laneId [[thread_index_in_simdgroup]]); \

instantiate_add_custom(64);
instantiate_add_custom(128);

}
