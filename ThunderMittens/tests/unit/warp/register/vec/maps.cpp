#include "maps.hpp"

#ifdef TEST_WARP_REGISTER_VEC_MAPS


template <typename T>
struct test_rv_exp {
    template<int W, int NW, rv_layout::all L> using valid = std::bool_constant<NW == 1 && W<=64>; // this is warp-level
    static inline const std::string test_identifier = std::is_same_v<T, bf16> ? "rv_exp=bf16" :
                                                      std::is_same_v<T, half> ? "rv_exp=half" :
                                                                                "rv_exp=float";
    static inline const std::string kernel_identifier = std::is_same_v<T, bf16> ? "rv_exp_bf16" :
                                                        std::is_same_v<T, half> ? "rv_exp_half" :
                                                                                  "rv_exp_float";
    using dtype = T;
    template<int W, int NW, rv_layout::all L>
    static void host_func(const std::vector<float> &i_ref, std::vector<float> &o_ref) {
        for(int i = 0; i < i_ref.size(); i++) {
            o_ref[i] = ::exp(i_ref[i]);
        }
    }
};

template <typename T>
struct test_rv_rv_add {
    template<int W, int NW, rv_layout::all L> using valid = std::bool_constant<NW == 1 && W<=64>; // this is warp-level
    static inline const std::string test_identifier = std::is_same_v<T, bf16> ? "rv_rv_add=bf16" :
                                                      std::is_same_v<T, half> ? "rv_rv_add=half" :
                                                                                "rv_rv_add=float";
    static inline const std::string kernel_identifier = std::is_same_v<T, bf16> ? "rv_rv_add_bf16" :
                                                        std::is_same_v<T, half> ? "rv_rv_add_half" :
                                                                                  "rv_rv_add_float";
    using dtype = T;
    template<int W, int NW, rv_layout::all L>
    static void host_func(const std::vector<float> &i_ref, std::vector<float> &o_ref) {
        for(int i = 0; i < i_ref.size(); i++) {
            o_ref[i] = i_ref[i] + i_ref[i];
        }
    }
};

template <typename T>
struct test_rv_t_add {
    template<int W, int NW, rv_layout::all L> using valid = std::bool_constant<NW == 1 && W<=16>; // this is warp-level
    static inline const std::string test_identifier = std::is_same_v<T, bf16> ? "rv_t_add=bf16" :
                                                      std::is_same_v<T, half> ? "rv_t_add=half" :
                                                                                "rv_t_add=float";
    static inline const std::string kernel_identifier = std::is_same_v<T, bf16> ? "rv_t_add_bf16" :
                                                        std::is_same_v<T, half> ? "rv_t_add_half" :
                                                                                  "rv_t_add_float";
    using dtype = T;
    template<int W, int NW, rv_layout::all L>
    static void host_func(const std::vector<float> &i_ref, std::vector<float> &o_ref) {
        for(int i = 0; i < i_ref.size(); i++) {
            o_ref[i] = i_ref[i] + 0.5f;
        }
    }
};


void warp::reg::vec::maps::tests(MTL::Device* device, MTL::CommandQueue* command_queue, test_data &results) {
    std::cout << "\n ----- Starting ops/warp/register/vec/maps tests! -----\n" << std::endl;
    constexpr int SIZE = INTENSITY_1 ? 2  :
                         INTENSITY_2 ? 4  :
                         INTENSITY_3 ? 8  :
                         INTENSITY_4 ? 16 : -1;
    
    general_sweep_size_1d_warp<test_rv_exp<float>, SIZE, rv_layout::align>::run(device, command_queue, results);
    general_sweep_size_1d_warp<test_rv_exp<float>, SIZE, rv_layout::ortho>::run(device, command_queue, results);
    general_sweep_size_1d_warp<test_rv_exp<float>, SIZE, rv_layout::naive>::run(device, command_queue, results);
    general_sweep_size_1d_warp<test_rv_exp<half>, SIZE, rv_layout::align>::run(device, command_queue, results);
    general_sweep_size_1d_warp<test_rv_exp<half>, SIZE, rv_layout::ortho>::run(device, command_queue, results);
    general_sweep_size_1d_warp<test_rv_exp<half>, SIZE, rv_layout::naive>::run(device, command_queue, results);
    general_sweep_size_1d_warp<test_rv_exp<bf16>, SIZE, rv_layout::align>::run(device, command_queue, results);
    general_sweep_size_1d_warp<test_rv_exp<bf16>, SIZE, rv_layout::ortho>::run(device, command_queue, results);
    general_sweep_size_1d_warp<test_rv_exp<bf16>, SIZE, rv_layout::naive>::run(device, command_queue, results);
    
    general_sweep_size_1d_warp<test_rv_rv_add<float>, SIZE, rv_layout::align>::run(device, command_queue, results);
    general_sweep_size_1d_warp<test_rv_rv_add<float>, SIZE, rv_layout::ortho>::run(device, command_queue, results);
    general_sweep_size_1d_warp<test_rv_rv_add<float>, SIZE, rv_layout::naive>::run(device, command_queue, results);
    general_sweep_size_1d_warp<test_rv_rv_add<half>, SIZE, rv_layout::align>::run(device, command_queue, results);
    general_sweep_size_1d_warp<test_rv_rv_add<half>, SIZE, rv_layout::ortho>::run(device, command_queue, results);
    general_sweep_size_1d_warp<test_rv_rv_add<half>, SIZE, rv_layout::naive>::run(device, command_queue, results);
    general_sweep_size_1d_warp<test_rv_rv_add<bf16>, SIZE, rv_layout::align>::run(device, command_queue, results);
    general_sweep_size_1d_warp<test_rv_rv_add<bf16>, SIZE, rv_layout::ortho>::run(device, command_queue, results);
    general_sweep_size_1d_warp<test_rv_rv_add<bf16>, SIZE, rv_layout::naive>::run(device, command_queue, results);
    
    general_sweep_size_1d_warp<test_rv_t_add<float>, SIZE, rv_layout::align>::run(device, command_queue, results);
    general_sweep_size_1d_warp<test_rv_t_add<float>, SIZE, rv_layout::ortho>::run(device, command_queue, results);
    general_sweep_size_1d_warp<test_rv_t_add<float>, SIZE, rv_layout::naive>::run(device, command_queue, results);
    general_sweep_size_1d_warp<test_rv_t_add<half>, SIZE, rv_layout::align>::run(device, command_queue, results);
    general_sweep_size_1d_warp<test_rv_t_add<half>, SIZE, rv_layout::ortho>::run(device, command_queue, results);
    general_sweep_size_1d_warp<test_rv_t_add<half>, SIZE, rv_layout::naive>::run(device, command_queue, results);
    general_sweep_size_1d_warp<test_rv_t_add<bf16>, SIZE, rv_layout::align>::run(device, command_queue, results);
    general_sweep_size_1d_warp<test_rv_t_add<bf16>, SIZE, rv_layout::ortho>::run(device, command_queue, results);
    general_sweep_size_1d_warp<test_rv_t_add<bf16>, SIZE, rv_layout::naive>::run(device, command_queue, results);
    
    

}

//
//int main() {
////    should_write_outputs = 1;
//    NS::Error** error;
//    MTL::CopyAllDevices();
//    MTL::Device* device = MTL::CreateSystemDefaultDevice();
//    MTL::CommandQueue* command_queue = device->newCommandQueue();
//    test_data results;
//    warp::reg::vec::maps::tests(device, command_queue, results);
//}
//

#endif

