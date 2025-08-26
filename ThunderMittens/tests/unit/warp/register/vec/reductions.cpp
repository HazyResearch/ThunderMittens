#include "reductions.hpp"

#ifdef TEST_WARP_REGISTER_VEC_REDUCTIONS


template <typename T>
struct test_rv_add {
    template<int W, int NW, rv_layout::all L> using valid = std::bool_constant<NW == 1 && W<=64>; // this is warp-level
    static inline const std::string test_identifier = std::is_same_v<T, bf16> ? "rv_add=bf16" :
                                                      std::is_same_v<T, half> ? "rv_add=half" :
                                                                                "rv_add=float";
    static inline const std::string kernel_identifier = std::is_same_v<T, bf16> ? "rv_add_bf16" :
                                                        std::is_same_v<T, half> ? "rv_add_half" :
                                                                                  "rv_add_float";
    using dtype = T;
    template<int W, int NW, rv_layout::all L>
    static void host_func(const std::vector<float> &i_ref, std::vector<float> &o_ref) {
        float sum = i_ref[0];
        float max = i_ref[0];
        for(int i = 1; i < i_ref.size(); i++) {
            sum = sum + i_ref[i];
            max = std::max(max, i_ref[i]);
//            o_ref[i] = 0.f;
        }
        for(int i = 0; i < i_ref.size(); i++) {
//            o_ref[i] = sum;
            o_ref[i] = max;
//            o_ref[i] = i_ref[i];
        }
    }
};


void warp::reg::vec::reductions::tests(MTL::Device* device, MTL::CommandQueue* command_queue, test_data &results) {
    std::cout << "\n ----- Starting ops/warp/register/vec/reductions tests! -----\n" << std::endl;
    constexpr int SIZE = INTENSITY_1 ? 2  :
                         INTENSITY_2 ? 4  :
                         INTENSITY_3 ? 8  :
                         INTENSITY_4 ? 16 : -1;
    
//    constexpr int SIZE = 16;
    
    general_sweep_size_1d_warp<test_rv_add<float>, SIZE, rv_layout::align>::run(device, command_queue, results);
    general_sweep_size_1d_warp<test_rv_add<float>, SIZE, rv_layout::ortho>::run(device, command_queue, results);
    general_sweep_size_1d_warp<test_rv_add<float>, SIZE, rv_layout::naive>::run(device, command_queue, results);
    general_sweep_size_1d_warp<test_rv_add<half>, SIZE, rv_layout::align>::run(device, command_queue, results);
    general_sweep_size_1d_warp<test_rv_add<half>, SIZE, rv_layout::ortho>::run(device, command_queue, results);
    general_sweep_size_1d_warp<test_rv_add<half>, SIZE, rv_layout::naive>::run(device, command_queue, results);
    general_sweep_size_1d_warp<test_rv_add<bf16>, SIZE, rv_layout::align>::run(device, command_queue, results);
    general_sweep_size_1d_warp<test_rv_add<bf16>, SIZE, rv_layout::ortho>::run(device, command_queue, results);
    general_sweep_size_1d_warp<test_rv_add<bf16>, SIZE, rv_layout::naive>::run(device, command_queue, results);

//    warp_wrapper_1d<test_rv_add<bf16>, 8, 1, rv_layout::naive>::run(device, command_queue, results);
    
    

}

//
//int main() {
////    should_write_outputs = 1;
//    NS::Error** error;
//    MTL::CopyAllDevices();
//    MTL::Device* device = MTL::CreateSystemDefaultDevice();
//    MTL::CommandQueue* command_queue = device->newCommandQueue();
//    test_data results;
//    warp::reg::vec::reductions::tests(device, command_queue, results);
//}


#endif

