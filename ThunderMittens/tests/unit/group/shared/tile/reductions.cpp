#include "reductions.hpp"

#ifdef TEST_GROUP_SHARED_TILE_REDUCTIONS
template <typename T>
struct test_group_row_add {
    template<int H, int W, int NW> using valid = std::bool_constant<W*H<=64>; // this is warp-level
    static inline const std::string test_identifier = std::is_same_v<T, bf16> ? "group_row_add=bf16" :
                                                      std::is_same_v<T, half> ? "group_row_add=half" :
                                                                                "group_row_add=float";
    static inline const std::string kernel_identifier = std::is_same_v<T, bf16> ? "group_row_add_bf16" :
                                                        std::is_same_v<T, half> ? "group_row_add_half" :
                                                                                  "group_row_add_float";
    using dtype = T;
    template<int H, int W, int NW>
    static void host_func(const std::vector<float> &i_ref, std::vector<float> &o_ref) {
        for (int y = 0; y < H * 8; y++) {
            float sum = 0.f;
            float max = -10.f;
            for (int x = 0; x < W * 8; x++) {
                sum += i_ref[y * W * 8 + x];
                max = std::max(max, i_ref[y * W * 8 + x]);
            }
            for (int x = 0; x < W * 8; x++) {
                o_ref[y * W * 8 + x] = sum;
//                o_ref[y * W * 8 + x] = i_ref[y * W * 8 + x];
                o_ref[y * W * 8 + x] = max;
            }
        }
    }
};

template <typename T>
struct test_group_col_add {
    template<int H, int W, int NW> using valid = std::bool_constant<W*H<=64>; // this is warp-level
    static inline const std::string test_identifier = std::is_same_v<T, bf16> ? "group_col_add=bf16" :
                                                      std::is_same_v<T, half> ? "group_col_add=half" :
                                                                                "group_col_add=float";
    static inline const std::string kernel_identifier = std::is_same_v<T, bf16> ? "group_col_add_bf16" :
                                                        std::is_same_v<T, half> ? "group_col_add_half" :
                                                                                  "group_col_add_float";
    using dtype = T;
    template<int H, int W, int NW>
    static void host_func(const std::vector<float> &i_ref, std::vector<float> &o_ref) {
        for (int x = 0; x < W * 8; x++) {
            float sum = 0.f;
            float max = -10.f;
            for (int y = 0; y < H * 8; y++) {
                sum += i_ref[y * W * 8 + x];
                max = std::max(max, i_ref[y * W * 8 + x]);
            }
            for (int y = 0; y < H * 8; y++) {
//                o_ref[y * W * 8 + x] = sum;
                o_ref[y * W * 8 + x] = max;
            }
        }
    }
};


void group::shared::tile::reductions::tests(MTL::Device* device, MTL::CommandQueue* command_queue, test_data &results) {
    std::cout << "\n ----- Starting ops/group/shared/tile/reductions tests! -----\n" << std::endl;
    constexpr int SIZE = INTENSITY_1 ? 2  :
                         INTENSITY_2 ? 4  :
                         INTENSITY_3 ? 8  :
                         INTENSITY_4 ? 16 : -1;
    small_general_sweep_size_2d<test_group_row_add<float>, SIZE, SIZE, 2>::run(device, command_queue, results);
    small_general_sweep_size_2d<test_group_row_add<float>, SIZE, SIZE, 3>::run(device, command_queue, results);
    small_general_sweep_size_2d<test_group_row_add<float>, SIZE, SIZE, 4>::run(device, command_queue, results);
    
    small_general_sweep_size_2d<test_group_row_add<half>,  SIZE, SIZE, 2>::run(device, command_queue, results);
    small_general_sweep_size_2d<test_group_row_add<half>,  SIZE, SIZE, 3>::run(device, command_queue, results);
    small_general_sweep_size_2d<test_group_row_add<half>,  SIZE, SIZE, 4>::run(device, command_queue, results);
    
    small_general_sweep_size_2d<test_group_row_add<bf16>,  SIZE, SIZE, 2>::run(device, command_queue, results);
    small_general_sweep_size_2d<test_group_row_add<bf16>,  SIZE, SIZE, 3>::run(device, command_queue, results);
    small_general_sweep_size_2d<test_group_row_add<bf16>,  SIZE, SIZE, 4>::run(device, command_queue, results);

    
    small_general_sweep_size_2d<test_group_col_add<float>, SIZE, SIZE, 2>::run(device, command_queue, results);
    small_general_sweep_size_2d<test_group_col_add<float>, SIZE, SIZE, 3>::run(device, command_queue, results);
    small_general_sweep_size_2d<test_group_col_add<float>, SIZE, SIZE, 4>::run(device, command_queue, results);
    
    small_general_sweep_size_2d<test_group_col_add<half>,  SIZE, SIZE, 2>::run(device, command_queue, results);
    small_general_sweep_size_2d<test_group_col_add<half>,  SIZE, SIZE, 3>::run(device, command_queue, results);
    small_general_sweep_size_2d<test_group_col_add<half>,  SIZE, SIZE, 4>::run(device, command_queue, results);
    
    small_general_sweep_size_2d<test_group_col_add<bf16>,  SIZE, SIZE, 2>::run(device, command_queue, results);
    small_general_sweep_size_2d<test_group_col_add<bf16>,  SIZE, SIZE, 3>::run(device, command_queue, results);
    small_general_sweep_size_2d<test_group_col_add<bf16>,  SIZE, SIZE, 4>::run(device, command_queue, results);

}

////
//int main() {
////    should_write_outputs = 1;
//    NS::Error** error;
//    MTL::CopyAllDevices();
//    MTL::Device* device = MTL::CreateSystemDefaultDevice();
//    MTL::CommandQueue* command_queue = device->newCommandQueue();
//    test_data results;
//    group::shared::tile::reductions::tests(device, command_queue, results);
//}


#endif

