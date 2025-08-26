#include "conversions.hpp"

#ifdef TEST_WARP_SHARED_TILE_CONVERSIONS
template <typename T>
struct test_st_copy {
    template<int H, int W, int NW> using valid = std::bool_constant<NW == 1 && W*H<=64>; // this is warp-level
    static inline const std::string test_identifier = std::is_same_v<T, bf16> ? "st_copy=bf16" :
                                                      std::is_same_v<T, half> ? "st_copy=half" :
                                                                                "st_copy=float";
    static inline const std::string kernel_identifier = std::is_same_v<T, bf16> ? "st_copy_bf16" :
                                                        std::is_same_v<T, half> ? "st_copy_half" :
                                                                                  "st_copy_float";
    using dtype = T;
    template<int H, int W, int NW>
    static void host_func(const std::vector<float> &i_ref, std::vector<float> &o_ref) {
        for(int i = 0; i < i_ref.size(); i++) {
            o_ref[i] = i_ref[i];
        }
    }
};




void warp::shared::tile::conversions::tests(MTL::Device* device, MTL::CommandQueue* command_queue, test_data &results) {
    std::cout << "\n ----- Starting ops/warp/shared/tile/conversions tests! -----\n" << std::endl;
    constexpr int SIZE = INTENSITY_1 ? 2  :
                         INTENSITY_2 ? 4  :
                         INTENSITY_3 ? 8  :
                         INTENSITY_4 ? 16 : -1;
    
    small_general_sweep_size_2d_warp<test_st_copy<float>, SIZE, SIZE>::run(device, command_queue, results);
    small_general_sweep_size_2d_warp<test_st_copy<half>,  SIZE, SIZE>::run(device, command_queue, results);
    small_general_sweep_size_2d_warp<test_st_copy<bf16>,  SIZE, SIZE>::run(device, command_queue, results);
}

//
//int main() {
////    should_write_outputs = 1;
//    NS::Error** error;
//    MTL::CopyAllDevices();
//    MTL::Device* device = MTL::CreateSystemDefaultDevice();
//    MTL::CommandQueue* command_queue = device->newCommandQueue();
//    test_data results;
//    warp::shared::tile::conversions::tests(device, command_queue, results);
//}


#endif

