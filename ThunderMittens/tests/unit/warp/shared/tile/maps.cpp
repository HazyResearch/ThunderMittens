#include "maps.hpp"

#ifdef TEST_WARP_SHARED_TILE_MAPS
template <typename T>
struct test_st_copy {
    template<int H, int W, int NW> using valid = std::bool_constant<NW == 1 && W*H<=64>; // this is warp-level
    static inline const std::string test_identifier = std::is_same_v<T, bf16> ? "st_exp=bf16" :
                                                      std::is_same_v<T, half> ? "st_exp=half" :
                                                                                "st_exp=float";
    static inline const std::string kernel_identifier = std::is_same_v<T, bf16> ? "st_exp_bf16" :
                                                        std::is_same_v<T, half> ? "st_exp_half" :
                                                                                  "st_exp_float";
    using dtype = T;
    template<int H, int W, int NW>
    static void host_func(const std::vector<float> &i_ref, std::vector<float> &o_ref) {
        for(int i = 0; i < i_ref.size(); i++) {
            o_ref[i] = ::exp(i_ref[i]);
        }
    }
};

template <typename T>
struct test_st_t_max {
    template<int H, int W, int NW> using valid = std::bool_constant<NW == 1 && W*H<=64>; // this is warp-level
    static inline const std::string test_identifier = std::is_same_v<T, bf16> ? "st_t_max=bf16" :
                                                      std::is_same_v<T, half> ? "st_t_max=half" :
                                                                                "st_t_max=float";
    static inline const std::string kernel_identifier = std::is_same_v<T, bf16> ? "st_t_max_bf16" :
                                                        std::is_same_v<T, half> ? "st_t_max_half" :
                                                                                  "st_t_max_float";
    using dtype = T;
    template<int H, int W, int NW>
    static void host_func(const std::vector<float> &i_ref, std::vector<float> &o_ref) {
        for(int i = 0; i < i_ref.size(); i++) {
            o_ref[i] = std::max(i_ref[i], 0.5f);
        }
    }
};

template <typename T>
struct test_st_st_add {
    template<int H, int W, int NW> using valid = std::bool_constant<NW == 1 && W*H<=64>; // this is warp-level
    static inline const std::string test_identifier = std::is_same_v<T, bf16> ? "st_st_add=bf16" :
                                                      std::is_same_v<T, half> ? "st_st_add=half" :
                                                                                "st_st_add=float";
    static inline const std::string kernel_identifier = std::is_same_v<T, bf16> ? "st_st_add_bf16" :
                                                        std::is_same_v<T, half> ? "st_st_add_half" :
                                                                                  "st_st_add_float";
    using dtype = T;
    template<int H, int W, int NW>
    static void host_func(const std::vector<float> &i_ref, std::vector<float> &o_ref) {
        for(int i = 0; i < i_ref.size(); i++) {
            o_ref[i] = i_ref[i] + i_ref[i];
        }
    }
};

template <typename T>
struct test_st_row_add {
    template<int H, int W, int NW> using valid = std::bool_constant<NW == 1 && W*H<=64>; // this is warp-level
    static inline const std::string test_identifier = std::is_same_v<T, bf16> ? "st_row_add=bf16" :
                                                      std::is_same_v<T, half> ? "st_row_add=half" :
                                                                                "st_row_add=float";
    static inline const std::string kernel_identifier = std::is_same_v<T, bf16> ? "st_row_add_bf16" :
                                                        std::is_same_v<T, half> ? "st_row_add_half" :
                                                                                  "st_row_add_float";
    using dtype = T;
    template<int H, int W, int NW>
    static void host_func(const std::vector<float> &i_ref, std::vector<float> &o_ref) {
        for (int y = 0; y < H * 8; y++) {
            for (int x = 0; x < W * 8; x++) {
                o_ref[y * W * 8 + x] = i_ref[y * W * 8 + x] + i_ref[y];
            }
        }
    }
};

template <typename T>
struct test_st_col_add {
    template<int H, int W, int NW> using valid = std::bool_constant<NW == 1 && W*H<=64>; // this is warp-level
    static inline const std::string test_identifier = std::is_same_v<T, bf16> ? "st_col_add=bf16" :
                                                      std::is_same_v<T, half> ? "st_col_add=half" :
                                                                                "st_col_add=float";
    static inline const std::string kernel_identifier = std::is_same_v<T, bf16> ? "st_col_add_bf16" :
                                                        std::is_same_v<T, half> ? "st_col_add_half" :
                                                                                  "st_col_add_float";
    using dtype = T;
    template<int H, int W, int NW>
    static void host_func(const std::vector<float> &i_ref, std::vector<float> &o_ref) {
        for (int x = 0; x < W * 8; x++) {
            for (int y = 0; y < H * 8; y++) {
                o_ref[y * W * 8 + x] = i_ref[y * W * 8 + x] + i_ref[x];
            }
        }
    }
};




void warp::shared::tile::maps::tests(MTL::Device* device, MTL::CommandQueue* command_queue, test_data &results) {
    std::cout << "\n ----- Starting ops/warp/shared/tile/maps tests! -----\n" << std::endl;
    constexpr int SIZE = INTENSITY_1 ? 2  :
                         INTENSITY_2 ? 4  :
                         INTENSITY_3 ? 8  :
                         INTENSITY_4 ? 16 : -1;
    
    small_general_sweep_size_2d_warp<test_st_copy<float>, SIZE, SIZE>::run(device, command_queue, results);
    small_general_sweep_size_2d_warp<test_st_copy<half>,  SIZE, SIZE>::run(device, command_queue, results);
    small_general_sweep_size_2d_warp<test_st_copy<bf16>,  SIZE, SIZE>::run(device, command_queue, results);

    small_general_sweep_size_2d_warp<test_st_t_max<float>, SIZE, SIZE>::run(device, command_queue, results);
    small_general_sweep_size_2d_warp<test_st_t_max<half>,  SIZE, SIZE>::run(device, command_queue, results);
    small_general_sweep_size_2d_warp<test_st_t_max<bf16>,  SIZE, SIZE>::run(device, command_queue, results);
    
    small_general_sweep_size_2d_warp<test_st_st_add<float>, SIZE, SIZE>::run(device, command_queue, results);
    small_general_sweep_size_2d_warp<test_st_st_add<half>,  SIZE, SIZE>::run(device, command_queue, results);
    small_general_sweep_size_2d_warp<test_st_st_add<bf16>,  SIZE, SIZE>::run(device, command_queue, results);
    
    small_general_sweep_size_2d_warp<test_st_row_add<float>, SIZE, SIZE>::run(device, command_queue, results);
    small_general_sweep_size_2d_warp<test_st_row_add<half>,  SIZE, SIZE>::run(device, command_queue, results);
    small_general_sweep_size_2d_warp<test_st_row_add<bf16>,  SIZE, SIZE>::run(device, command_queue, results);
    
    small_general_sweep_size_2d_warp<test_st_col_add<float>, SIZE, SIZE>::run(device, command_queue, results);
    small_general_sweep_size_2d_warp<test_st_col_add<half>,  SIZE, SIZE>::run(device, command_queue, results);
    small_general_sweep_size_2d_warp<test_st_col_add<bf16>,  SIZE, SIZE>::run(device, command_queue, results);
}

//int main() {
////    should_write_outputs = 1;
//    NS::Error** error;
//    MTL::CopyAllDevices();
//    MTL::Device* device = MTL::CreateSystemDefaultDevice();
//    MTL::CommandQueue* command_queue = device->newCommandQueue();
//    test_data results;
//    warp::shared::tile::maps::tests(device, command_queue, results);
//}


#endif

