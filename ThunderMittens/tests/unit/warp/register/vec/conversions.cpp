#include "conversions.hpp"

#ifdef TEST_WARP_REGISTER_VEC_CONVERSIONS


template <typename T>
struct test_rv_copy {
    template<int L, int NW, rv_layout::all L1, rv_layout::all L2> using valid = std::bool_constant<NW == 1 && L<=64>; // this is warp-level
    static inline const std::string test_identifier = std::is_same_v<T, bf16> ? "rv_copy=bf16" :
                                                      std::is_same_v<T, half> ? "rv_copy=half" :
                                                                                "rv_copy=float";
    static inline const std::string kernel_identifier = std::is_same_v<T, bf16> ? "rv_copy_bf16" :
                                                        std::is_same_v<T, half> ? "rv_copy_half" :
                                                                                  "rv_copy_float";
    using dtype = T;
    template<int L, int NW, rv_layout::all L1, rv_layout::all L2>
    static void host_func(const std::vector<float> &i_ref, std::vector<float> &o_ref) {
        for(int i = 0; i < i_ref.size(); i++) {
            o_ref[i] = i_ref[i];
        }
    }
};


void warp::reg::vec::conversions::tests(MTL::Device* device, MTL::CommandQueue* command_queue, test_data &results) {
    std::cout << "\n ----- Starting ops/warp/register/tile/maps tests! -----\n" << std::endl;
    constexpr int SIZE = INTENSITY_1 ? 2  :
                         INTENSITY_2 ? 4  :
                         INTENSITY_3 ? 8  :
                         INTENSITY_4 ? 16 : -1;
    
    general_sweep_size_1d_warp<test_rv_copy<float>, SIZE, rv_layout::align, rv_layout::align>::run(device, command_queue, results);
    general_sweep_size_1d_warp<test_rv_copy<float>, SIZE, rv_layout::ortho, rv_layout::ortho>::run(device, command_queue, results);
    general_sweep_size_1d_warp<test_rv_copy<float>, SIZE, rv_layout::naive, rv_layout::naive>::run(device, command_queue, results);
    
    general_sweep_size_1d_warp<test_rv_copy<float>, SIZE, rv_layout::align, rv_layout::ortho>::run(device, command_queue, results);
    general_sweep_size_1d_warp<test_rv_copy<float>, SIZE, rv_layout::ortho, rv_layout::align>::run(device, command_queue, results);
    
    general_sweep_size_1d_warp<test_rv_copy<float>, SIZE, rv_layout::align, rv_layout::naive>::run(device, command_queue, results);
    general_sweep_size_1d_warp<test_rv_copy<float>, SIZE, rv_layout::naive, rv_layout::align>::run(device, command_queue, results);
    
    general_sweep_size_1d_warp<test_rv_copy<float>, SIZE, rv_layout::ortho, rv_layout::naive>::run(device, command_queue, results);
    general_sweep_size_1d_warp<test_rv_copy<float>, SIZE, rv_layout::naive, rv_layout::ortho>::run(device, command_queue, results);
    
    
    general_sweep_size_1d_warp<test_rv_copy<half>, SIZE, rv_layout::align, rv_layout::align>::run(device, command_queue, results);
    general_sweep_size_1d_warp<test_rv_copy<half>, SIZE, rv_layout::ortho, rv_layout::ortho>::run(device, command_queue, results);
    general_sweep_size_1d_warp<test_rv_copy<half>, SIZE, rv_layout::naive, rv_layout::naive>::run(device, command_queue, results);
    
    general_sweep_size_1d_warp<test_rv_copy<half>, SIZE, rv_layout::align, rv_layout::ortho>::run(device, command_queue, results);
    general_sweep_size_1d_warp<test_rv_copy<half>, SIZE, rv_layout::ortho, rv_layout::align>::run(device, command_queue, results);
    
    general_sweep_size_1d_warp<test_rv_copy<half>, SIZE, rv_layout::align, rv_layout::naive>::run(device, command_queue, results);
    general_sweep_size_1d_warp<test_rv_copy<half>, SIZE, rv_layout::naive, rv_layout::align>::run(device, command_queue, results);
    
    general_sweep_size_1d_warp<test_rv_copy<half>, SIZE, rv_layout::ortho, rv_layout::naive>::run(device, command_queue, results);
    general_sweep_size_1d_warp<test_rv_copy<half>, SIZE, rv_layout::naive, rv_layout::ortho>::run(device, command_queue, results);
    
    
    general_sweep_size_1d_warp<test_rv_copy<bf16>, SIZE, rv_layout::align, rv_layout::align>::run(device, command_queue, results);
    general_sweep_size_1d_warp<test_rv_copy<bf16>, SIZE, rv_layout::ortho, rv_layout::ortho>::run(device, command_queue, results);
    general_sweep_size_1d_warp<test_rv_copy<bf16>, SIZE, rv_layout::naive, rv_layout::naive>::run(device, command_queue, results);
    
    general_sweep_size_1d_warp<test_rv_copy<bf16>, SIZE, rv_layout::align, rv_layout::ortho>::run(device, command_queue, results);
    general_sweep_size_1d_warp<test_rv_copy<bf16>, SIZE, rv_layout::ortho, rv_layout::align>::run(device, command_queue, results);
    
    general_sweep_size_1d_warp<test_rv_copy<bf16>, SIZE, rv_layout::align, rv_layout::naive>::run(device, command_queue, results);
    general_sweep_size_1d_warp<test_rv_copy<bf16>, SIZE, rv_layout::naive, rv_layout::align>::run(device, command_queue, results);
    
    general_sweep_size_1d_warp<test_rv_copy<bf16>, SIZE, rv_layout::ortho, rv_layout::naive>::run(device, command_queue, results);
    general_sweep_size_1d_warp<test_rv_copy<bf16>, SIZE, rv_layout::naive, rv_layout::ortho>::run(device, command_queue, results);
    

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


