#include "maps.hpp"

#ifdef TEST_GROUP_SHARED_VEC_MAPS
template <typename T>
struct test_group_sv_exp {
    template<int L, int NW> using valid = std::bool_constant<L<=64>; // this is warp-level
    static inline const std::string test_identifier = std::is_same_v<T, bf16> ? "group_sv_exp=bf16" :
                                                      std::is_same_v<T, half> ? "group_sv_exp=half" :
                                                                                "group_sv_exp=float";
    static inline const std::string kernel_identifier = std::is_same_v<T, bf16> ? "group_sv_exp_bf16" :
                                                        std::is_same_v<T, half> ? "group_sv_exp_half" :
                                                                                  "group_sv_exp_float";
    using dtype = T;
    template<int L, int NW>
    static void host_func(const std::vector<float> &i_ref, std::vector<float> &o_ref) {
        for (int i = 0; i < i_ref.size(); i++) {
            o_ref[i] = ::exp(i_ref[i]);
        }
    }
};

template <typename T>
struct test_group_sv_sv_add {
    template<int L, int NW> using valid = std::bool_constant<L<=64>; // this is warp-level
    static inline const std::string test_identifier = std::is_same_v<T, bf16> ? "group_sv_sv_add=bf16" :
                                                      std::is_same_v<T, half> ? "group_sv_sv_add=half" :
                                                                                "group_sv_sv_add=float";
    static inline const std::string kernel_identifier = std::is_same_v<T, bf16> ? "group_sv_sv_add_bf16" :
                                                        std::is_same_v<T, half> ? "group_sv_sv_add_half" :
                                                                                  "group_sv_sv_add_float";
    using dtype = T;
    template<int L, int NW>
    static void host_func(const std::vector<float> &i_ref, std::vector<float> &o_ref) {
        for (int i = 0; i < i_ref.size(); i++) {
            o_ref[i] = i_ref[i] + i_ref[i];
        }
    }
};

template <typename T>
struct test_group_sv_t_add {
    template<int L, int NW> using valid = std::bool_constant<L<=64>; // this is warp-level
    static inline const std::string test_identifier = std::is_same_v<T, bf16> ? "group_sv_t_add=bf16" :
                                                      std::is_same_v<T, half> ? "group_sv_t_add=half" :
                                                                                "group_sv_t_add=float";
    static inline const std::string kernel_identifier = std::is_same_v<T, bf16> ? "group_sv_t_add_bf16" :
                                                        std::is_same_v<T, half> ? "group_sv_t_add_half" :
                                                                                  "group_sv_t_add_float";
    using dtype = T;
    template<int L, int NW>
    static void host_func(const std::vector<float> &i_ref, std::vector<float> &o_ref) {
        for (int i = 0; i < i_ref.size(); i++) {
            o_ref[i] = i_ref[i] + 0.5f;
        }
    }
};


void group::shared::vec::maps::tests(MTL::Device* device, MTL::CommandQueue* command_queue, test_data &results) {
    std::cout << "\n ----- Starting ops/group/shared/vec/maps tests! -----\n" << std::endl;
    constexpr int SIZE = INTENSITY_1 ? 2  :
                         INTENSITY_2 ? 4  :
                         INTENSITY_3 ? 8  :
                         INTENSITY_4 ? 16 : -1;
    general_sweep_size_1d<test_group_sv_exp<float>, SIZE, 2>::run(device, command_queue, results);
    general_sweep_size_1d<test_group_sv_exp<float>, SIZE, 3>::run(device, command_queue, results);
    general_sweep_size_1d<test_group_sv_exp<float>, SIZE, 4>::run(device, command_queue, results);
    general_sweep_size_1d<test_group_sv_exp<half>,  SIZE, 2>::run(device, command_queue, results);
    general_sweep_size_1d<test_group_sv_exp<half>,  SIZE, 3>::run(device, command_queue, results);
    general_sweep_size_1d<test_group_sv_exp<half>,  SIZE, 4>::run(device, command_queue, results);
    general_sweep_size_1d<test_group_sv_exp<bf16>,  SIZE, 2>::run(device, command_queue, results);
    general_sweep_size_1d<test_group_sv_exp<bf16>,  SIZE, 3>::run(device, command_queue, results);
    general_sweep_size_1d<test_group_sv_exp<bf16>,  SIZE, 4>::run(device, command_queue, results);

    
    general_sweep_size_1d<test_group_sv_sv_add<float>, SIZE, 2>::run(device, command_queue, results);
    general_sweep_size_1d<test_group_sv_sv_add<float>, SIZE, 3>::run(device, command_queue, results);
    general_sweep_size_1d<test_group_sv_sv_add<float>, SIZE, 4>::run(device, command_queue, results);
    general_sweep_size_1d<test_group_sv_sv_add<half>,  SIZE, 2>::run(device, command_queue, results);
    general_sweep_size_1d<test_group_sv_sv_add<half>,  SIZE, 3>::run(device, command_queue, results);
    general_sweep_size_1d<test_group_sv_sv_add<half>,  SIZE, 4>::run(device, command_queue, results);
    general_sweep_size_1d<test_group_sv_sv_add<bf16>,  SIZE, 2>::run(device, command_queue, results);
    general_sweep_size_1d<test_group_sv_sv_add<bf16>,  SIZE, 3>::run(device, command_queue, results);
    general_sweep_size_1d<test_group_sv_sv_add<bf16>,  SIZE, 4>::run(device, command_queue, results);
    
    
    general_sweep_size_1d<test_group_sv_t_add<float>, SIZE, 2>::run(device, command_queue, results);
    general_sweep_size_1d<test_group_sv_t_add<float>, SIZE, 3>::run(device, command_queue, results);
    general_sweep_size_1d<test_group_sv_t_add<float>, SIZE, 4>::run(device, command_queue, results);
    general_sweep_size_1d<test_group_sv_t_add<half>,  SIZE, 2>::run(device, command_queue, results);
    general_sweep_size_1d<test_group_sv_t_add<half>,  SIZE, 3>::run(device, command_queue, results);
    general_sweep_size_1d<test_group_sv_t_add<half>,  SIZE, 4>::run(device, command_queue, results);
    general_sweep_size_1d<test_group_sv_t_add<bf16>,  SIZE, 2>::run(device, command_queue, results);
    general_sweep_size_1d<test_group_sv_t_add<bf16>,  SIZE, 3>::run(device, command_queue, results);
    general_sweep_size_1d<test_group_sv_t_add<bf16>,  SIZE, 4>::run(device, command_queue, results);
    //test_group_sv_t_add
}

//int main() {
////    should_write_outputs = 1;
//    NS::Error** error;
//    MTL::CopyAllDevices();
//    MTL::Device* device = MTL::CreateSystemDefaultDevice();
//    MTL::CommandQueue* command_queue = device->newCommandQueue();
//    test_data results;
//    group::shared::vec::maps::tests(device, command_queue, results);
//}


#endif

