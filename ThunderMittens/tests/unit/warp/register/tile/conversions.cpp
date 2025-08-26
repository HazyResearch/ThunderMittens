#include "conversions.hpp"

#ifdef TEST_WARP_REGISTER_TILE_CONVERSIONS
template<typename test, int H, int W, int NUM_WORKERS, typename... args>
struct tranpose_wrapper_2d {
    using dtype = gmem_dtype<test>; // defaults to bf16 in global memory if the test doesn't specify.
    
    static void run(MTL::Device* device, MTL::CommandQueue* command_queue, test_data& results) {
        test_info this_result;
        this_result.label       = generate_test_name<false, H,W,NUM_WORKERS, args...>(test::test_identifier);
        this_result.kernel_name = generate_test_name<true, H,W,NUM_WORKERS, args...>(test::kernel_identifier);
        if constexpr (test::template valid<H, W, NUM_WORKERS, args...>::value) {
            constexpr int B = 1, D = 1, R = 1, C = 1;
            constexpr int SIZE = H*W*8*8 * B * D * R * C;
            MTL::Buffer* d_i;
            MTL::Buffer* d_o;
            std::vector<float> i_ref(SIZE);
            std::vector<float> o_ref(SIZE);
            initialize<dtype>(device, &d_i, &d_o, i_ref, o_ref);
            NS::String* kernel_name = NS::String::string(this_result.kernel_name.c_str(), NS::ASCIIStringEncoding);
            
            run_kernel<NUM_WORKERS>(device, command_queue, kernel_name, d_i, d_o);
            test::template host_func<H, W, NUM_WORKERS, args...>(i_ref, o_ref);
            this_result.result = validate<dtype>(d_i, d_o, i_ref, o_ref, this_result.label, W*8);
            
        }
        else {
            this_result.result = test_result::INVALID;
        }
        results.push_back(this_result);
    }
};

template <typename T>
struct test_rt_swap_layout {
    template<int H, int W, int NW, rt_layout::all L> using valid = std::bool_constant<NW == 1 && W*H<=64>; // this is warp-level
    static inline const std::string test_identifier = std::is_same_v<T, bf16> ? "rt_swap_layout=bf16" :
                                                      std::is_same_v<T, half> ? "rt_swap_layout=half" :
                                                                                "rt_swap_layout=float";
    static inline const std::string kernel_identifier = std::is_same_v<T, bf16> ? "rt_swap_layout_bf16" :
                                                        std::is_same_v<T, half> ? "rt_swap_layout_half" :
                                                                                  "rt_swap_layout_float";
    using dtype = T;
    template<int H, int W, int NW, rt_layout::all L>
    static void host_func(const std::vector<float> &i_ref, std::vector<float> &o_ref) {
        for(int i = 0; i < i_ref.size(); i++) {
            o_ref[i] = i_ref[i];
        }
    }
};


template <typename T>
struct test_rt_transpose {
    template<int H, int W, int NW, rt_layout::all L> using valid = std::bool_constant<NW == 1 && W*H<=64>; // this is warp-level
    static inline const std::string test_identifier = std::is_same_v<T, bf16> ? "rt_transpose=bf16" :
                                                      std::is_same_v<T, half> ? "rt_transpose=half" :
                                                                                "rt_transpose=float";
    static inline const std::string kernel_identifier = std::is_same_v<T, bf16> ? "rt_transpose_bf16" :
                                                        std::is_same_v<T, half> ? "rt_transpose_half" :
                                                                                  "rt_transpose_float";
    using dtype = T;
    template<int H, int W, int NW, rt_layout::all L>
    static void host_func(const std::vector<float> &i_ref, std::vector<float> &o_ref) {
        for (int y = 0; y < W * 8; y++) {
            for (int x = 0; x < H * 8; x++) {
                o_ref[y * H*8 + x] = i_ref[x * W * 8 + y];
            }
        }
    }
};

template <typename T>
struct test_rt_transpose_inplace {
    template<int H, int W, int NW, rt_layout::all L> using valid = std::bool_constant<NW == 1 && W*H<=64 && H == W>; // this is warp-level
    static inline const std::string test_identifier = std::is_same_v<T, bf16> ? "rt_transpose_inplace=bf16" :
                                                      std::is_same_v<T, half> ? "rt_transpose_inplace=half" :
                                                                                "rt_transpose_inplace=float";
    static inline const std::string kernel_identifier = std::is_same_v<T, bf16> ? "rt_transpose_inplace_bf16" :
                                                        std::is_same_v<T, half> ? "rt_transpose_inplace_half" :
                                                                                  "rt_transpose_inplace_float";
    using dtype = T;
    template<int H, int W, int NW, rt_layout::all L>
    static void host_func(const std::vector<float> &i_ref, std::vector<float> &o_ref) {
        for (int y = 0; y < W * 8; y++) {
            for (int x = 0; x < H * 8; x++) {
                o_ref[y * H*8 + x] = i_ref[x * W * 8 + y];
            }
        }
    }
};

template <typename T>
struct test_rt_make_causal {
    template<int H, int W, int NW, rt_layout::all L> using valid = std::bool_constant<NW == 1 && W*H<=64 && H == W>; // this is warp-level
    static inline const std::string test_identifier = std::is_same_v<T, bf16> ? "rt_make_causal=bf16" :
                                                      std::is_same_v<T, half> ? "rt_make_causal=half" :
                                                                                "rt_make_causal=float";
    static inline const std::string kernel_identifier = std::is_same_v<T, bf16> ? "rt_make_causal_bf16" :
                                                        std::is_same_v<T, half> ? "rt_make_causal_half" :
                                                                                  "rt_make_causal_float";
    using dtype = T;
    template<int H, int W, int NW, rt_layout::all L>
    static void host_func(const std::vector<float> &i_ref, std::vector<float> &o_ref) {
        for (int y = 0; y < H * 8; y++) {
            for (int x = 0; x < W * 8; x++) {
                if (y < x) {
                    o_ref[y * W * 8 + x] = 0.f;
                } else {
                    o_ref[y * W * 8 + x] = i_ref[y * W * 8 + x];
                }
                
            }
        }
    }
};



void warp::reg::tile::conversions::tests(MTL::Device* device, MTL::CommandQueue* command_queue, test_data &results) {
    std::cout << "\n ----- Starting ops/warp/register/tile/maps tests! -----\n" << std::endl;
    constexpr int SIZE = INTENSITY_1 ? 2  :
                         INTENSITY_2 ? 4  :
                         INTENSITY_3 ? 8  :
                         INTENSITY_4 ? 16 : -1;
    
    small_general_sweep_size_2d_warp<test_rt_swap_layout<float>, SIZE, SIZE, rt_layout::row>::run(device, command_queue, results);
    small_general_sweep_size_2d_warp<test_rt_swap_layout<float>, SIZE, SIZE, rt_layout::col>::run(device, command_queue, results);
    small_general_sweep_size_2d_warp<test_rt_swap_layout<half>,  SIZE, SIZE, rt_layout::row>::run(device, command_queue, results);
    small_general_sweep_size_2d_warp<test_rt_swap_layout<half>,  SIZE, SIZE, rt_layout::col>::run(device, command_queue, results);
    small_general_sweep_size_2d_warp<test_rt_swap_layout<bf16>,  SIZE, SIZE, rt_layout::row>::run(device, command_queue, results);
    small_general_sweep_size_2d_warp<test_rt_swap_layout<bf16>,  SIZE, SIZE, rt_layout::col>::run(device, command_queue, results);
    
    small_general_sweep_size_2d_warp<test_rt_transpose<float>, SIZE, SIZE, rt_layout::row>::run(device, command_queue, results);
    small_general_sweep_size_2d_warp<test_rt_transpose<float>, SIZE, SIZE, rt_layout::col>::run(device, command_queue, results);
    small_general_sweep_size_2d_warp<test_rt_transpose<half>,  SIZE, SIZE, rt_layout::row>::run(device, command_queue, results);
    small_general_sweep_size_2d_warp<test_rt_transpose<half>,  SIZE, SIZE, rt_layout::col>::run(device, command_queue, results);
    small_general_sweep_size_2d_warp<test_rt_transpose<bf16>,  SIZE, SIZE, rt_layout::row>::run(device, command_queue, results);
    small_general_sweep_size_2d_warp<test_rt_transpose<bf16>,  SIZE, SIZE, rt_layout::col>::run(device, command_queue, results);
    
    small_general_sweep_size_2d_warp<test_rt_transpose_inplace<float>, SIZE, SIZE, rt_layout::row>::run(device, command_queue, results);
    small_general_sweep_size_2d_warp<test_rt_transpose_inplace<float>, SIZE, SIZE, rt_layout::col>::run(device, command_queue, results);
    small_general_sweep_size_2d_warp<test_rt_transpose_inplace<half>,  SIZE, SIZE, rt_layout::row>::run(device, command_queue, results);
    small_general_sweep_size_2d_warp<test_rt_transpose_inplace<half>,  SIZE, SIZE, rt_layout::col>::run(device, command_queue, results);
    small_general_sweep_size_2d_warp<test_rt_transpose_inplace<bf16>,  SIZE, SIZE, rt_layout::row>::run(device, command_queue, results);
    small_general_sweep_size_2d_warp<test_rt_transpose_inplace<bf16>,  SIZE, SIZE, rt_layout::col>::run(device, command_queue, results);
    
    small_general_sweep_size_2d_warp<test_rt_make_causal<float>, SIZE, SIZE, rt_layout::row>::run(device, command_queue, results);
    small_general_sweep_size_2d_warp<test_rt_make_causal<float>, SIZE, SIZE, rt_layout::col>::run(device, command_queue, results);
    small_general_sweep_size_2d_warp<test_rt_make_causal<half>,  SIZE, SIZE, rt_layout::row>::run(device, command_queue, results);
    small_general_sweep_size_2d_warp<test_rt_make_causal<half>,  SIZE, SIZE, rt_layout::col>::run(device, command_queue, results);
    small_general_sweep_size_2d_warp<test_rt_make_causal<bf16>,  SIZE, SIZE, rt_layout::row>::run(device, command_queue, results);
    small_general_sweep_size_2d_warp<test_rt_make_causal<bf16>,  SIZE, SIZE, rt_layout::col>::run(device, command_queue, results);
}

//
//int main() {
////    should_write_outputs = 1;
//    NS::Error** error;
//    MTL::CopyAllDevices();
//    MTL::Device* device = MTL::CreateSystemDefaultDevice();
//    MTL::CommandQueue* command_queue = device->newCommandQueue();
//    test_data results;
//    warp::reg::tile::conversions::tests(device, command_queue, results);
//}


#endif

