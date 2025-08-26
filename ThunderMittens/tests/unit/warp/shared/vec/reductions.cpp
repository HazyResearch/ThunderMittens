#include "reductions.hpp"

#ifdef TEST_WARP_SHARED_VEC_REDUCTIONS
template <typename T>
struct test_t_sv_max {
    template<int L, int NW> using valid = std::bool_constant<NW == 1 && L<=64>; // this is warp-level
    static inline const std::string test_identifier = std::is_same_v<T, bf16> ? "t_sv_max=bf16" :
                                                      std::is_same_v<T, half> ? "t_sv_max=half" :
                                                                                "t_sv_max=float";
    static inline const std::string kernel_identifier = std::is_same_v<T, bf16> ? "t_sv_max_bf16" :
                                                        std::is_same_v<T, half> ? "t_sv_max_half" :
                                                                                  "t_sv_max_float";
    using dtype = T;
    template<int L, int NW>
    static void host_func(const std::vector<float> &i_ref, std::vector<float> &o_ref) {
        float val = 0.f;
        for (int i = 0; i < i_ref.size(); i++) {
            val = std::max(val, i_ref[i]);
        }
        for (int i = 0; i < i_ref.size(); i++) {
            o_ref[i] = val;
//            o_ref[i] = i_ref[i];
        }
    }
};



void warp::shared::vec::reductions::tests(MTL::Device* device, MTL::CommandQueue* command_queue, test_data &results) {
    std::cout << "\n ----- Starting ops/warp/shared/vec/reductions tests! -----\n" << std::endl;
    constexpr int SIZE = INTENSITY_1 ? 2  :
                         INTENSITY_2 ? 4  :
                         INTENSITY_3 ? 8  :
                         INTENSITY_4 ? 16 : -1;
    
    general_sweep_size_1d_warp<test_t_sv_max<float>, SIZE>::run(device, command_queue, results);
    general_sweep_size_1d_warp<test_t_sv_max<half>,  SIZE>::run(device, command_queue, results);
    general_sweep_size_1d_warp<test_t_sv_max<bf16>,  SIZE>::run(device, command_queue, results);
    
    //sv_t_add
}

//int main() {
////    should_write_outputs = 1;
//    NS::Error** error;
//    MTL::CopyAllDevices();
//    MTL::Device* device = MTL::CreateSystemDefaultDevice();
//    MTL::CommandQueue* command_queue = device->newCommandQueue();
//    test_data results;
//    warp::shared::vec::reductions::tests(device, command_queue, results);
//}


#endif

