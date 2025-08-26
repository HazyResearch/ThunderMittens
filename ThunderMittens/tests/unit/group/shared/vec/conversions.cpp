#include "conversions.hpp"

#ifdef TEST_GROUP_SHARED_VEC_CONVERSIONS
template <typename T>
struct test_group_sv_copy {
    template<int L, int NW> using valid = std::bool_constant<L<=64>; // this is warp-level
    static inline const std::string test_identifier = std::is_same_v<T, bf16> ? "group_sv_copy=bf16" :
                                                      std::is_same_v<T, half> ? "group_sv_copy=half" :
                                                                                "group_sv_copy=float";
    static inline const std::string kernel_identifier = std::is_same_v<T, bf16> ? "group_sv_copy_bf16" :
                                                        std::is_same_v<T, half> ? "group_sv_copy_half" :
                                                                                  "group_sv_copy_float";
    using dtype = T;
    template<int L, int NW>
    static void host_func(const std::vector<float> &i_ref, std::vector<float> &o_ref) {
        for (int i = 0; i < i_ref.size(); i++) {
            o_ref[i] = i_ref[i];
        }
    }
};


void group::shared::vec::conversions::tests(MTL::Device* device, MTL::CommandQueue* command_queue, test_data &results) {
    std::cout << "\n ----- Starting ops/group/shared/vec/conversions tests! -----\n" << std::endl;
    constexpr int SIZE = INTENSITY_1 ? 2  :
                         INTENSITY_2 ? 4  :
                         INTENSITY_3 ? 8  :
                         INTENSITY_4 ? 16 : -1;
    general_sweep_size_1d<test_group_sv_copy<float>, SIZE, 2>::run(device, command_queue, results);
    general_sweep_size_1d<test_group_sv_copy<float>, SIZE, 3>::run(device, command_queue, results);
    general_sweep_size_1d<test_group_sv_copy<float>, SIZE, 4>::run(device, command_queue, results);
    
    general_sweep_size_1d<test_group_sv_copy<half>,  SIZE, 2>::run(device, command_queue, results);
    general_sweep_size_1d<test_group_sv_copy<half>,  SIZE, 3>::run(device, command_queue, results);
    general_sweep_size_1d<test_group_sv_copy<half>,  SIZE, 4>::run(device, command_queue, results);
    
    general_sweep_size_1d<test_group_sv_copy<bf16>,  SIZE, 2>::run(device, command_queue, results);
    general_sweep_size_1d<test_group_sv_copy<bf16>,  SIZE, 3>::run(device, command_queue, results);
    general_sweep_size_1d<test_group_sv_copy<bf16>,  SIZE, 4>::run(device, command_queue, results);

}

//int main() {
////    should_write_outputs = 1;
//    NS::Error** error;
//    MTL::CopyAllDevices();
//    MTL::Device* device = MTL::CreateSystemDefaultDevice();
//    MTL::CommandQueue* command_queue = device->newCommandQueue();
//    test_data results;
//    group::shared::vec::conversions::tests(device, command_queue, results);
//}


#endif

