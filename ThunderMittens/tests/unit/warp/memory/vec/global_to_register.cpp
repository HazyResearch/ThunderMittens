#include "global_to_register.hpp"
#ifdef TEST_WARP_MEMORY_VEC_GLOBAL_TO_REGISTER
#include "MetalSingle.hpp"



template<typename T>
struct reg_vec_loadstore {
    using dtype = T;
    template<int S, int NW, typename... args> using valid = std::bool_constant<NW == 1 && S <= 64>; // this is warp-level
    
    
    static inline const std::string test_identifier = std::is_same_v<T, bf16> ? "reg_vec_loadstore_gmem=bf16" :
                                                      std::is_same_v<T, half> ? "reg_vec_loadstore_gmem=half" :
                                                                                "reg_vec_loadstore_gmem=float";
    static inline const std::string kernel_identifier = std::is_same_v<T, bf16> ? "reg_vec_loadstore_gmem_bf16" :
                                                        std::is_same_v<T, half> ? "reg_vec_loadstore_gmem_half" :
                                                                                  "reg_vec_loadstore_gmem_float";
    template<int S, int NW, rv_layout::all L>
    static void host_func(const std::vector<float> &i_ref, std::vector<float> &o_ref) {
        o_ref = i_ref; // overwrite the whole thing
    }
};

void warp::memory::vec::global_to_register::tests(MTL::Device* device, MTL::CommandQueue* command_queue, test_data &results) {
    std::cout << "\n ----- Starting ops/warp/memory/vec/global_to_register tests! -----\n" << std::endl;
    constexpr int SIZE = INTENSITY_1 ? 2  :
                         INTENSITY_2 ? 4  :
                         INTENSITY_3 ? 8  :
                         INTENSITY_4 ? 16 : -1;
                         
    general_sweep_size_1d_warp<reg_vec_loadstore<float>, SIZE, rv_layout::align>::run(device, command_queue, results);
    general_sweep_size_1d_warp<reg_vec_loadstore<float>, SIZE, rv_layout::ortho>::run(device, command_queue, results);
    general_sweep_size_1d_warp<reg_vec_loadstore<float>, SIZE, rv_layout::naive>::run(device, command_queue, results);
    general_sweep_size_1d_warp<reg_vec_loadstore<bf16>,  SIZE, rv_layout::align>::run(device, command_queue, results);
    general_sweep_size_1d_warp<reg_vec_loadstore<bf16>,  SIZE, rv_layout::ortho>::run(device, command_queue, results);
    general_sweep_size_1d_warp<reg_vec_loadstore<bf16>,  SIZE, rv_layout::naive>::run(device, command_queue, results);
    general_sweep_size_1d_warp<reg_vec_loadstore<half>,  SIZE, rv_layout::align>::run(device, command_queue, results);
    general_sweep_size_1d_warp<reg_vec_loadstore<half>,  SIZE, rv_layout::ortho>::run(device, command_queue, results);
    general_sweep_size_1d_warp<reg_vec_loadstore<half>,  SIZE, rv_layout::naive>::run(device, command_queue, results);
}

#endif

//
//
//int main() {
////    should_write_outputs = 1;
//    NS::Error** error;
//    MTL::CopyAllDevices();
//    MTL::Device* device = MTL::CreateSystemDefaultDevice();
//    MTL::CommandQueue* command_queue = device->newCommandQueue();
//    test_data results;
//    warp::memory::vec::global_to_register::tests(device, command_queue, results);
//}
