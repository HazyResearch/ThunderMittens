#include "maps.hpp"

#ifdef TEST_WARP_REGISTER_TILE_MAPS

template <typename T>
struct test_rt_exp {
    template<int H, int W, int NW, rt_layout::all L> using valid = std::bool_constant<NW == 1 && W*H<=64>; // this is warp-level
    static inline const std::string test_identifier = std::is_same_v<T, bf16> ? "rt_exp=bf16" :
                                                      std::is_same_v<T, half> ? "rt_exp=half" :
                                                                                "rt_exp=float";
    static inline const std::string kernel_identifier = std::is_same_v<T, bf16> ? "rt_exp_bf16" :
                                                        std::is_same_v<T, half> ? "rt_exp_half" :
                                                                                  "rt_exp_float";
    using dtype = T;
    template<int H, int W, int NW, rt_layout::all L>
    static void host_func(const std::vector<float> &i_ref, std::vector<float> &o_ref) {
        for(int i = 0; i < i_ref.size(); i++) {
            o_ref[i] = ::expf(i_ref[i]);
        }
    }
};

template <typename T>
struct test_rt_t_max {
    template<int H, int W, int NW, rt_layout::all L> using valid = std::bool_constant<NW == 1 && W*H<=64>; // this is warp-level
    static inline const std::string test_identifier = std::is_same_v<T, bf16> ? "rt_t_max=bf16" :
                                                      std::is_same_v<T, half> ? "rt_t_max=half" :
                                                                                "rt_t_max=float";
    static inline const std::string kernel_identifier = std::is_same_v<T, bf16> ? "rt_t_max_bf16" :
                                                        std::is_same_v<T, half> ? "rt_t_max_half" :
                                                                                  "rt_t_max_float";
    using dtype = T;
    template<int H, int W, int NW, rt_layout::all L>
    static void host_func(const std::vector<float> &i_ref, std::vector<float> &o_ref) {
        float val = 0.f;
        for(int i = 0; i < i_ref.size(); i++) {
            o_ref[i] = std::max(i_ref[i], val);
        }
    }
};


template <typename T>
struct test_rt_rt_add {
    template<int H, int W, int NW, rt_layout::all L> using valid = std::bool_constant<NW == 1 && W*H<=64>; // this is warp-level
    static inline const std::string test_identifier = std::is_same_v<T, bf16> ? "rt_rt_add=bf16" :
                                                      std::is_same_v<T, half> ? "rt_rt_add=half" :
                                                                                "rt_rt_add=float";
    static inline const std::string kernel_identifier = std::is_same_v<T, bf16> ? "rt_rt_add_bf16" :
                                                        std::is_same_v<T, half> ? "rt_rt_add_half" :
                                                                                  "rt_rt_add_float";
    using dtype = T;
    template<int H, int W, int NW, rt_layout::all L>
    static void host_func(const std::vector<float> &i_ref, std::vector<float> &o_ref) {
        for(int i = 0; i < i_ref.size(); i++) {
            o_ref[i] = i_ref[i] + i_ref[i];
        }
    }
};


template <typename T>
struct test_rt_v_row_add {
    template<int H, int W, int NW, rt_layout::all L> using valid = std::bool_constant<NW == 1 && W*H<=64>; // this is warp-level
    static inline const std::string test_identifier = std::is_same_v<T, bf16> ? "rt_v_row_add=bf16" :
                                                      std::is_same_v<T, half> ? "rt_v_row_add=half" :
                                                                                "rt_v_row_add=float";
    static inline const std::string kernel_identifier = std::is_same_v<T, bf16> ? "rt_v_row_add_bf16" :
                                                        std::is_same_v<T, half> ? "rt_v_row_add_half" :
                                                                                  "rt_v_row_add_float";
    using dtype = T;
    template<int H, int W, int NW, rt_layout::all L>
    static void host_func(const std::vector<float> &i_ref, std::vector<float> &o_ref) {
        for (int y = 0; y < H*8; y++) {
            for (int x = 0; x < W*8; x++) {
                float val = i_ref[y];
                o_ref[y * W*8 + x] = i_ref[y * W*8 + x] + val;
            }
        }
    }
};

template <typename T>
struct test_rt_v_col_add {
    template<int H, int W, int NW, rt_layout::all L> using valid = std::bool_constant<NW == 1 && W*H<=64>; // this is warp-level
    static inline const std::string test_identifier = std::is_same_v<T, bf16> ? "rt_v_col_add=bf16" :
                                                      std::is_same_v<T, half> ? "rt_v_col_add=half" :
                                                                                "rt_v_col_add=float";
    static inline const std::string kernel_identifier = std::is_same_v<T, bf16> ? "rt_v_col_add_bf16" :
                                                        std::is_same_v<T, half> ? "rt_v_col_add_half" :
                                                                                  "rt_v_col_add_float";
    using dtype = T;
    template<int H, int W, int NW, rt_layout::all L>
    static void host_func(const std::vector<float> &i_ref, std::vector<float> &o_ref) {
        for (int y = 0; y < H*8; y++) {
            for (int x = 0; x < W*8; x++) {
                float val = i_ref[x];
                o_ref[y * W*8 + x] = i_ref[y * W*8 + x] + val;
            }
        }
    }
};




void warp::reg::tile::maps::tests(MTL::Device* device, MTL::CommandQueue* command_queue, test_data &results) {
    std::cout << "\n ----- Starting ops/warp/register/tile/maps tests! -----\n" << std::endl;
    constexpr int SIZE = INTENSITY_1 ? 2  :
                         INTENSITY_2 ? 4  :
                         INTENSITY_3 ? 8  :
                         INTENSITY_4 ? 16 : -1;
    
//    constexpr int SIZE = 16;
    
    small_general_sweep_size_2d_warp<test_rt_exp<float>, SIZE, SIZE, rt_layout::row>::run(device, command_queue, results);
    small_general_sweep_size_2d_warp<test_rt_exp<float>, SIZE, SIZE, rt_layout::col>::run(device, command_queue, results);
    small_general_sweep_size_2d_warp<test_rt_exp<half>,  SIZE, SIZE, rt_layout::row>::run(device, command_queue, results);
    small_general_sweep_size_2d_warp<test_rt_exp<half>,  SIZE, SIZE, rt_layout::col>::run(device, command_queue, results);
    small_general_sweep_size_2d_warp<test_rt_exp<bf16>,  SIZE, SIZE, rt_layout::row>::run(device, command_queue, results);
    small_general_sweep_size_2d_warp<test_rt_exp<bf16>,  SIZE, SIZE, rt_layout::col>::run(device, command_queue, results);

    small_general_sweep_size_2d_warp<test_rt_t_max<float>, SIZE, SIZE, rt_layout::row>::run(device, command_queue, results);
    small_general_sweep_size_2d_warp<test_rt_t_max<float>, SIZE, SIZE, rt_layout::col>::run(device, command_queue, results);
    small_general_sweep_size_2d_warp<test_rt_t_max<half>,  SIZE, SIZE, rt_layout::row>::run(device, command_queue, results);
    small_general_sweep_size_2d_warp<test_rt_t_max<half>,  SIZE, SIZE, rt_layout::col>::run(device, command_queue, results);
    small_general_sweep_size_2d_warp<test_rt_t_max<bf16>,  SIZE, SIZE, rt_layout::row>::run(device, command_queue, results);
    small_general_sweep_size_2d_warp<test_rt_t_max<bf16>,  SIZE, SIZE, rt_layout::col>::run(device, command_queue, results);

    small_general_sweep_size_2d_warp<test_rt_rt_add<float>, SIZE, SIZE, rt_layout::row>::run(device, command_queue, results);
    small_general_sweep_size_2d_warp<test_rt_rt_add<float>, SIZE, SIZE, rt_layout::col>::run(device, command_queue, results);
    small_general_sweep_size_2d_warp<test_rt_rt_add<half>,  SIZE, SIZE, rt_layout::row>::run(device, command_queue, results);
    small_general_sweep_size_2d_warp<test_rt_rt_add<half>,  SIZE, SIZE, rt_layout::col>::run(device, command_queue, results);
    small_general_sweep_size_2d_warp<test_rt_rt_add<bf16>,  SIZE, SIZE, rt_layout::row>::run(device, command_queue, results);
    small_general_sweep_size_2d_warp<test_rt_rt_add<bf16>,  SIZE, SIZE, rt_layout::col>::run(device, command_queue, results);
    
    small_general_sweep_size_2d_warp<test_rt_v_row_add<float>, SIZE, SIZE, rt_layout::row>::run(device, command_queue, results);
    small_general_sweep_size_2d_warp<test_rt_v_row_add<float>, SIZE, SIZE, rt_layout::col>::run(device, command_queue, results);
    small_general_sweep_size_2d_warp<test_rt_v_row_add<half>,  SIZE, SIZE, rt_layout::row>::run(device, command_queue, results);
    small_general_sweep_size_2d_warp<test_rt_v_row_add<half>,  SIZE, SIZE, rt_layout::col>::run(device, command_queue, results);
    small_general_sweep_size_2d_warp<test_rt_v_row_add<bf16>,  SIZE, SIZE, rt_layout::row>::run(device, command_queue, results);
    small_general_sweep_size_2d_warp<test_rt_v_row_add<bf16>,  SIZE, SIZE, rt_layout::col>::run(device, command_queue, results);
    
    small_general_sweep_size_2d_warp<test_rt_v_col_add<float>, SIZE, SIZE, rt_layout::row>::run(device, command_queue, results);
    small_general_sweep_size_2d_warp<test_rt_v_col_add<float>, SIZE, SIZE, rt_layout::col>::run(device, command_queue, results);
    small_general_sweep_size_2d_warp<test_rt_v_col_add<half>,  SIZE, SIZE, rt_layout::row>::run(device, command_queue, results);
    small_general_sweep_size_2d_warp<test_rt_v_col_add<half>,  SIZE, SIZE, rt_layout::col>::run(device, command_queue, results);
    small_general_sweep_size_2d_warp<test_rt_v_col_add<bf16>,  SIZE, SIZE, rt_layout::row>::run(device, command_queue, results);
    small_general_sweep_size_2d_warp<test_rt_v_col_add<bf16>,  SIZE, SIZE, rt_layout::col>::run(device, command_queue, results);
    
}

//
//int main() {
////    should_write_outputs = 1;
//    NS::Error** error;
//    MTL::CopyAllDevices();
//    MTL::Device* device = MTL::CreateSystemDefaultDevice();
//    MTL::CommandQueue* command_queue = device->newCommandQueue();
//    test_data results;
//    warp::reg::tile::maps::tests(device, command_queue, results);
//}


#endif

