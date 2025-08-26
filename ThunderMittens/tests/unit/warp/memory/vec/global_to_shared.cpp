#include "global_to_shared.hpp"
#ifdef TEST_WARP_MEMORY_VEC_GLOBAL_TO_SHARED
#include "MetalSingle.hpp"



template<typename T>
struct shared_vec_loadstore {
    using dtype = T;
    template<int S, int NW, typename... args> using valid = std::bool_constant<NW == 1 && S <= 64>; // this is warp-level
    
    
    static inline const std::string test_identifier = std::is_same_v<T, bf16> ? "shared_vec_loadstore_gmem=bf16" :
                                                      std::is_same_v<T, half> ? "shared_vec_loadstore_gmem=half" :
                                                                                "shared_vec_loadstore_gmem=float";
    static inline const std::string kernel_identifier = std::is_same_v<T, bf16> ? "shared_vec_loadstore_gmem_bf16" :
                                                        std::is_same_v<T, half> ? "shared_vec_loadstore_gmem_half" :
                                                                                  "shared_vec_loadstore_gmem_float";
    template<int S, int NW>
    static void host_func(const std::vector<float> &i_ref, std::vector<float> &o_ref) {
        o_ref = i_ref; // overwrite the whole thing
    }
};

void warp::memory::vec::global_to_shared::tests(MTL::Device* device, MTL::CommandQueue* command_queue, test_data &results) {
    std::cout << "\n ----- Starting ops/warp/memory/vec/global_to_shared tests! -----\n" << std::endl;
    constexpr int SIZE = INTENSITY_1 ? 2  :
                         INTENSITY_2 ? 4  :
                         INTENSITY_3 ? 8  :
                         INTENSITY_4 ? 16 : -1;
//    constexpr int SIZE = 16;
                         
    general_sweep_size_1d_warp<shared_vec_loadstore<float>, SIZE>::run(device, command_queue, results);
    general_sweep_size_1d_warp<shared_vec_loadstore<bf16>,  SIZE>::run(device, command_queue, results);
    general_sweep_size_1d_warp<shared_vec_loadstore<half>,  SIZE>::run(device, command_queue, results);
}

#endif

//
//
//int main() {
//    should_write_outputs = 1;
//    NS::Error** error;
//    MTL::CopyAllDevices();
//    MTL::Device* device = MTL::CreateSystemDefaultDevice();
//    MTL::CommandQueue* command_queue = device->newCommandQueue();
//    test_data results;
//    warp::memory::vec::global_to_shared::tests(device, command_queue, results);
//}
