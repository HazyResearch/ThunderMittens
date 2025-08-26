#include "global_to_register.hpp"
#ifdef TEST_WARP_MEMORY_TILE_GLOBAL_TO_REGISTER
#include "MetalSingle.hpp"



template<typename T>
struct load_store {
    using dtype = T;
    template<int H, int W, int NW, typename... args> using valid = std::bool_constant<NW == 1 && W*H <= 64>; // this is warp-level
    
    
    static inline const std::string test_identifier = std::is_same_v<T, bf16> ? "reg_loadstore_gmem=bf16" :
                                                      std::is_same_v<T, half> ? "reg_loadstore_gmem=half" :
                                                                                "reg_loadstore_gmem=float";
    static inline const std::string kernel_identifier = std::is_same_v<T, bf16> ? "reg_loadstore_gmem_bf16" :
                                                        std::is_same_v<T, half> ? "reg_loadstore_gmem_half" :
                                                                                  "reg_loadstore_gmem_float";
    template<int H, int W, int NW, rt_layout::all L>
    static void host_func(const std::vector<float> &i_ref, std::vector<float> &o_ref) {
        o_ref = i_ref; // overwrite the whole thing
    }
};

void warp::memory::tile::global_to_register::tests(MTL::Device* device, MTL::CommandQueue* command_queue, test_data &results) {
    std::cout << "\n ----- Starting ops/warp/memory/tile/global_to_register tests! -----\n" << std::endl;
//    constexpr int SIZE = INTENSITY_1 ? 2  :
//                         INTENSITY_2 ? 4  :
//                         INTENSITY_3 ? 8  :
//                         INTENSITY_4 ? 16 : -1;
    constexpr int SIZE = 1;
                         
    general_sweep_size_2d_warp<load_store<float>, SIZE, SIZE, rt_layout::row>::run(device, command_queue, results);
    general_sweep_size_2d_warp<load_store<float>, SIZE, SIZE, rt_layout::col>::run(device, command_queue, results);
    general_sweep_size_2d_warp<load_store<bf16>,  SIZE, SIZE, rt_layout::row>::run(device, command_queue, results);
    general_sweep_size_2d_warp<load_store<bf16>,  SIZE, SIZE, rt_layout::col>::run(device, command_queue, results);
    general_sweep_size_2d_warp<load_store<half>,  SIZE, SIZE, rt_layout::row>::run(device, command_queue, results);
    general_sweep_size_2d_warp<load_store<half>,  SIZE, SIZE, rt_layout::col>::run(device, command_queue, results);
}

#endif


//
//int main() {
////    should_write_outputs = 1;
//    NS::Error** error;
//    MTL::CopyAllDevices();
//    MTL::Device* device = MTL::CreateSystemDefaultDevice();
//    MTL::CommandQueue* command_queue = device->newCommandQueue();
//    test_data results;
//    warp::memory::tile::global_to_register::tests(device, command_queue, results);
//}
