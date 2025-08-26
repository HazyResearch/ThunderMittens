#include "tk.metal"
#include <metal_stdlib>
namespace mittens {

#define PARAMS(T) \
    device T* y [[buffer(1)]], \
    device T* x [[buffer(0)]], \
    device T* out [[buffer(2)]], \
    constant const int& ndim1 [[buffer(3)]], \
    constant const int& ndim2 [[buffer(4)]], \
    ushort3 index [[thread_position_in_grid]], \
    ushort3 num_threads [[threads_per_grid]], \
    uint3 threadgroup_id [[threadgroup_position_in_grid]], \
    ushort laneId [[thread_index_in_simdgroup]]

template <typename T>
[[kernel]] void add_rt(PARAMS(T)) {
    
    // using global_layout = gl<T, 1, 1, 8, 8>;
    // global_layout gl_x(x,   nullptr, nullptr, nullptr, nullptr);
    // global_layout gl_y(y,   nullptr, nullptr, nullptr, nullptr);
    // global_layout gl_o(out, nullptr, nullptr, nullptr, nullptr);

    using global_layout = gl<T, 1, 1, -1, -1>;
    global_layout gl_x(x,   nullptr, nullptr, ndim1, ndim2);
    global_layout gl_y(y,   nullptr, nullptr, ndim1, ndim2);
    global_layout gl_o(out, nullptr, nullptr, ndim1, ndim2);

    rt<T,8, 8> reg_x, reg_y, reg_z;
    int y_idx = threadgroup_id.y;
    int x_idx = threadgroup_id.x; 
    // load(reg_x, gl_x, {0,0,(int)threadgroup_id.y, (int)threadgroup_id.x}, laneId);
    // load(reg_y, gl_y, {0,0,(int)threadgroup_id.y, (int)threadgroup_id.x}, laneId);
    load(reg_x, gl_x, {0,0, y_idx, x_idx}, laneId);
    load(reg_y, gl_y, {0,0, y_idx, x_idx}, laneId);
    load(reg_z, gl_y, {0,0, y_idx, x_idx}, laneId);

    // add(reg_y, reg_x, reg_x);
    mma_AB(reg_y, reg_x, reg_z, reg_y);
    mma_AB(reg_y, reg_x, reg_z, reg_y);
    mma_AB(reg_y, reg_x, reg_z, reg_y);
    mma_AB(reg_y, reg_x, reg_z, reg_y);
    
    // store(gl_o, reg_y, {0,0,(int)threadgroup_id.y, (int)threadgroup_id.x}, laneId);
    store(gl_o, reg_y, {0,0,y_idx, x_idx}, laneId);
    // out[laneId] = 5;
}

#define instantiate_add_custom(type_name, T)                           \
  template [[host_name("add_rt_" #type_name)]] [[kernel]] void         \
  add_rt<T>(PARAMS(T));

instantiate_add_custom(float32, float);
instantiate_add_custom(float16, half);
instantiate_add_custom(bfloat16, bf16);

}