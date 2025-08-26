#include "reductions.hpp"

#ifdef TEST_WARP_SHARED_TILE_REDUCTIONS


template <typename T>
struct test_st_row_max {
    template<int H, int W, int NW> using valid = std::bool_constant<NW == 1 && W*H<=64>; // this is warp-level
    static inline const std::string test_identifier = std::is_same_v<T, bf16> ? "st_row_max=bf16" :
                                                      std::is_same_v<T, half> ? "st_row_max=half" :
                                                                                "st_row_max=float";
    static inline const std::string kernel_identifier = std::is_same_v<T, bf16> ? "st_row_max_bf16" :
                                                        std::is_same_v<T, half> ? "st_row_max_half" :
                                                                                  "st_row_max_float";
    using dtype = T;
    template<int H, int W, int NW>
    static void host_func(const std::vector<float> &i_ref, std::vector<float> &o_ref) {
        for (int y = 0; y < H * 8; y++) {
            float val = 0.f;
            for (int x = 0; x < W * 8; x++) {
                val = std::max(val, i_ref[y * W * 8 + x]);
            }
            for (int x = 0; x < W * 8; x++) {
                o_ref[y * W * 8 + x] = val;
            }
        }
    }
};

template <typename T>
struct test_st_col_max {
    template<int H, int W, int NW> using valid = std::bool_constant<NW == 1 && W*H<=64>; // this is warp-level
    static inline const std::string test_identifier = std::is_same_v<T, bf16> ? "st_col_max=bf16" :
                                                      std::is_same_v<T, half> ? "st_col_max=half" :
                                                                                "st_col_max=float";
    static inline const std::string kernel_identifier = std::is_same_v<T, bf16> ? "st_col_max_bf16" :
                                                        std::is_same_v<T, half> ? "st_col_max_half" :
                                                                                  "st_col_max_float";
    using dtype = T;
    template<int H, int W, int NW>
    static void host_func(const std::vector<float> &i_ref, std::vector<float> &o_ref) {
        for (int x = 0; x < W * 8; x++) {
            float val = 0.f;
            for (int y = 0; y < H * 8; y++) {
                val = std::max(val, i_ref[y * W * 8 + x]);
            }
            for (int y = 0; y < H * 8; y++) {
                o_ref[y * W * 8 + x] = val;
            }
        }
    }
};


void warp::shared::tile::reductions::tests(MTL::Device* device, MTL::CommandQueue* command_queue, test_data &results) {
    std::cout << "\n ----- Starting ops/warp/shared/tile/reductions tests! -----\n" << std::endl;
    constexpr int SIZE = INTENSITY_1 ? 2  :
                         INTENSITY_2 ? 4  :
                         INTENSITY_3 ? 8  :
                         INTENSITY_4 ? 16 : -1;
    
    small_general_sweep_size_2d_warp<test_st_row_max<float>, SIZE, SIZE>::run(device, command_queue, results);
    small_general_sweep_size_2d_warp<test_st_row_max<half>,  SIZE, SIZE>::run(device, command_queue, results);
    small_general_sweep_size_2d_warp<test_st_row_max<bf16>,  SIZE, SIZE>::run(device, command_queue, results);
    
    small_general_sweep_size_2d_warp<test_st_col_max<float>, SIZE, SIZE>::run(device, command_queue, results);
    small_general_sweep_size_2d_warp<test_st_col_max<half>,  SIZE, SIZE>::run(device, command_queue, results);
    small_general_sweep_size_2d_warp<test_st_col_max<bf16>,  SIZE, SIZE>::run(device, command_queue, results);

}

//int main() {
////    should_write_outputs = 1;
//    NS::Error** error;
//    MTL::CopyAllDevices();
//    MTL::Device* device = MTL::CreateSystemDefaultDevice();
//    MTL::CommandQueue* command_queue = device->newCommandQueue();
//    test_data results;
//    warp::shared::tile::reductions::tests(device, command_queue, results);
//}


#endif

