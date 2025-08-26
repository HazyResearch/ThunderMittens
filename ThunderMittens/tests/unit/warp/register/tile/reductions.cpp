#include "reductions.hpp"

#ifdef TEST_WARP_REGISTER_TILE_REDUCTIONS

template <typename T>
struct test_reg_row_reduce {
    template<int H, int W, int NW, rt_layout::all L> using valid = std::bool_constant<NW == 1 && W*H<=64>; // this is warp-level
    static inline const std::string test_identifier = std::is_same_v<T, bf16> ? "reg_row_reduce=bf16" :
                                                      std::is_same_v<T, half> ? "reg_row_reduce=half" :
                                                                                "reg_row_reduce=float";
    static inline const std::string kernel_identifier = std::is_same_v<T, bf16> ? "reg_row_reduce_bf16" :
                                                        std::is_same_v<T, half> ? "reg_row_reduce_half" :
                                                                                  "reg_row_reduce_float";
    using dtype = T;
    template<int H, int W, int NW, rt_layout::all L>
    static void host_func(const std::vector<float> &i_ref, std::vector<float> &o_ref) {
        for (int y = 0; y < H * 8; y++) {
            float max_val = 0.f;
            for (int x = 0; x < W * 8; x++) {
                max_val = std::max(max_val, i_ref[y * W * 8 + x]);
            }
            for (int x = 0; x < W * 8; x++) {
                o_ref[y * W * 8 + x] = max_val;
            }
        }
    }
};

template <typename T>
struct test_reg_col_reduce {
    template<int H, int W, int NW, rt_layout::all L> using valid = std::bool_constant<NW == 1 && W*H<=64>; // this is warp-level
    static inline const std::string test_identifier = std::is_same_v<T, bf16> ? "reg_col_reduce=bf16" :
                                                      std::is_same_v<T, half> ? "reg_col_reduce=half" :
                                                                                "reg_col_reduce=float";
    static inline const std::string kernel_identifier = std::is_same_v<T, bf16> ? "reg_col_reduce_bf16" :
                                                        std::is_same_v<T, half> ? "reg_col_reduce_half" :
                                                                                  "reg_col_reduce_float";
    using dtype = T;
    template<int H, int W, int NW, rt_layout::all L>
    static void host_func(const std::vector<float> &i_ref, std::vector<float> &o_ref) {
        for (int x = 0; x < W * 8; x++) {
            float max_val = 0.f;
            for (int y = 0; y < H * 8; y++) {
                max_val = std::max(max_val, i_ref[y * W * 8 + x]);
            }
            for (int y = 0; y < H * 8; y++) {
                o_ref[y * W * 8 + x] = max_val;
            }
        }
    }
};





void warp::reg::tile::reductions::tests(MTL::Device* device, MTL::CommandQueue* command_queue, test_data &results) {
    std::cout << "\n ----- Starting ops/warp/register/tile/reductions tests! -----\n" << std::endl;
    constexpr int SIZE = INTENSITY_1 ? 2  :
                         INTENSITY_2 ? 4  :
                         INTENSITY_3 ? 8  :
                         INTENSITY_4 ? 16 : -1;
    
//    constexpr int SIZE = 16;
    
    small_general_sweep_size_2d_warp<test_reg_row_reduce<float>, SIZE, SIZE, rt_layout::row>::run(device, command_queue, results);
    small_general_sweep_size_2d_warp<test_reg_row_reduce<float>, SIZE, SIZE, rt_layout::col>::run(device, command_queue, results);
    small_general_sweep_size_2d_warp<test_reg_row_reduce<half>,  SIZE, SIZE, rt_layout::row>::run(device, command_queue, results);
    small_general_sweep_size_2d_warp<test_reg_row_reduce<half>,  SIZE, SIZE, rt_layout::col>::run(device, command_queue, results);
    small_general_sweep_size_2d_warp<test_reg_row_reduce<bf16>,  SIZE, SIZE, rt_layout::row>::run(device, command_queue, results);
    small_general_sweep_size_2d_warp<test_reg_row_reduce<bf16>,  SIZE, SIZE, rt_layout::col>::run(device, command_queue, results);
    
    small_general_sweep_size_2d_warp<test_reg_col_reduce<float>, SIZE, SIZE, rt_layout::row>::run(device, command_queue, results);
    small_general_sweep_size_2d_warp<test_reg_col_reduce<float>, SIZE, SIZE, rt_layout::col>::run(device, command_queue, results);
    small_general_sweep_size_2d_warp<test_reg_col_reduce<half>,  SIZE, SIZE, rt_layout::row>::run(device, command_queue, results);
    small_general_sweep_size_2d_warp<test_reg_col_reduce<half>,  SIZE, SIZE, rt_layout::col>::run(device, command_queue, results);
    small_general_sweep_size_2d_warp<test_reg_col_reduce<bf16>,  SIZE, SIZE, rt_layout::row>::run(device, command_queue, results);
    small_general_sweep_size_2d_warp<test_reg_col_reduce<bf16>,  SIZE, SIZE, rt_layout::col>::run(device, command_queue, results);

}


//int main() {
////    should_write_outputs = 1;
//    NS::Error** error;
//    MTL::CopyAllDevices();
//    MTL::Device* device = MTL::CreateSystemDefaultDevice();
//    MTL::CommandQueue* command_queue = device->newCommandQueue();
//    test_data results;
//    warp::reg::tile::reductions::tests(device, command_queue, results);
//}


#endif

