#include "maps.hpp"

#ifdef TEST_GROUP_SHARED_TILE_MAPS
template <typename T>
struct test_group_st_exp {
    template<int H, int W, int NW> using valid = std::bool_constant<W*H<=64>; // this is warp-level
    static inline const std::string test_identifier = std::is_same_v<T, bf16> ? "group_st_exp=bf16" :
                                                      std::is_same_v<T, half> ? "group_st_exp=half" :
                                                                                "group_st_exp=float";
    static inline const std::string kernel_identifier = std::is_same_v<T, bf16> ? "group_st_exp_bf16" :
                                                        std::is_same_v<T, half> ? "group_st_exp_half" :
                                                                                  "group_st_exp_float";
    using dtype = T;
    template<int H, int W, int NW>
    static void host_func(const std::vector<float> &i_ref, std::vector<float> &o_ref) {
        for(int i = 0; i < i_ref.size(); i++) {
            o_ref[i] = ::exp(i_ref[i]);
        }
    }
};

template <typename T>
struct test_group_st_st_add {
    template<int H, int W, int NW> using valid = std::bool_constant<W*H<=64>; // this is warp-level
    static inline const std::string test_identifier = std::is_same_v<T, bf16> ? "group_st_st_add=bf16" :
                                                      std::is_same_v<T, half> ? "group_st_st_add=half" :
                                                                                "group_st_st_add=float";
    static inline const std::string kernel_identifier = std::is_same_v<T, bf16> ? "group_st_st_add_bf16" :
                                                        std::is_same_v<T, half> ? "group_st_st_add_half" :
                                                                                  "group_st_st_add_float";
    using dtype = T;
    template<int H, int W, int NW>
    static void host_func(const std::vector<float> &i_ref, std::vector<float> &o_ref) {
        for(int i = 0; i < i_ref.size(); i++) {
            o_ref[i] = i_ref[i] + i_ref[i];
        }
    }
};

template <typename T>
struct test_group_st_t_add {
    template<int H, int W, int NW> using valid = std::bool_constant<W*H<=64>; // this is warp-level
    static inline const std::string test_identifier = std::is_same_v<T, bf16> ? "group_st_t_add=bf16" :
                                                      std::is_same_v<T, half> ? "group_st_t_add=half" :
                                                                                "group_st_t_add=float";
    static inline const std::string kernel_identifier = std::is_same_v<T, bf16> ? "group_st_t_add_bf16" :
                                                        std::is_same_v<T, half> ? "group_st_t_add_half" :
                                                                                  "group_st_t_add_float";
    using dtype = T;
    template<int H, int W, int NW>
    static void host_func(const std::vector<float> &i_ref, std::vector<float> &o_ref) {
        for(int i = 0; i < i_ref.size(); i++) {
            o_ref[i] = i_ref[i] + 0.5f;
        }
    }
};

template <typename T>
struct test_group_st_sv_row_add {
    template<int H, int W, int NW> using valid = std::bool_constant<W*H<=64>; // this is warp-level
    static inline const std::string test_identifier = std::is_same_v<T, bf16> ? "group_st_sv_row_add=bf16" :
                                                      std::is_same_v<T, half> ? "group_st_sv_row_add=half" :
                                                                                "group_st_sv_row_add=float";
    static inline const std::string kernel_identifier = std::is_same_v<T, bf16> ? "group_st_sv_row_add_bf16" :
                                                        std::is_same_v<T, half> ? "group_st_sv_row_add_half" :
                                                                                  "group_st_sv_row_add_float";
    using dtype = T;
    template<int H, int W, int NW>
    static void host_func(const std::vector<float> &i_ref, std::vector<float> &o_ref) {
        for (int y = 0; y < H * 8; y++) {
            for (int x = 0; x < W * 8; x++) {
                o_ref[y * W * 8 + x] = i_ref[y * W * 8 + x] + i_ref[y];
            }
        }
    }
};

template <typename T>
struct test_group_st_sv_col_add {
    template<int H, int W, int NW> using valid = std::bool_constant<W*H<=64>; // this is warp-level
    static inline const std::string test_identifier = std::is_same_v<T, bf16> ? "group_st_sv_col_add=bf16" :
                                                      std::is_same_v<T, half> ? "group_st_sv_col_add=half" :
                                                                                "group_st_sv_col_add=float";
    static inline const std::string kernel_identifier = std::is_same_v<T, bf16> ? "group_st_sv_col_add_bf16" :
                                                        std::is_same_v<T, half> ? "group_st_sv_col_add_half" :
                                                                                  "group_st_sv_col_add_float";
    using dtype = T;
    template<int H, int W, int NW>
    static void host_func(const std::vector<float> &i_ref, std::vector<float> &o_ref) {
        for (int y = 0; y < H * 8; y++) {
            for (int x = 0; x < W * 8; x++) {
                o_ref[y * W * 8 + x] = i_ref[y * W * 8 + x] + i_ref[x];
            }
        }
    }
};



void group::shared::tile::maps::tests(MTL::Device* device, MTL::CommandQueue* command_queue, test_data &results) {
    std::cout << "\n ----- Starting ops/group/shared/tile/maps tests! -----\n" << std::endl;
    constexpr int SIZE = INTENSITY_1 ? 2  :
                         INTENSITY_2 ? 4  :
                         INTENSITY_3 ? 8  :
                         INTENSITY_4 ? 16 : -1;
    small_general_sweep_size_2d<test_group_st_exp<float>, SIZE, SIZE, 2>::run(device, command_queue, results);
    small_general_sweep_size_2d<test_group_st_exp<float>, SIZE, SIZE, 3>::run(device, command_queue, results);
    small_general_sweep_size_2d<test_group_st_exp<float>, SIZE, SIZE, 4>::run(device, command_queue, results);
    
    small_general_sweep_size_2d<test_group_st_exp<half>,  SIZE, SIZE, 2>::run(device, command_queue, results);
    small_general_sweep_size_2d<test_group_st_exp<half>,  SIZE, SIZE, 3>::run(device, command_queue, results);
    small_general_sweep_size_2d<test_group_st_exp<half>,  SIZE, SIZE, 4>::run(device, command_queue, results);
    
    small_general_sweep_size_2d<test_group_st_exp<bf16>,  SIZE, SIZE, 2>::run(device, command_queue, results);
    small_general_sweep_size_2d<test_group_st_exp<bf16>,  SIZE, SIZE, 3>::run(device, command_queue, results);
    small_general_sweep_size_2d<test_group_st_exp<bf16>,  SIZE, SIZE, 4>::run(device, command_queue, results);
    
    
    small_general_sweep_size_2d<test_group_st_st_add<float>, SIZE, SIZE, 2>::run(device, command_queue, results);
    small_general_sweep_size_2d<test_group_st_st_add<float>, SIZE, SIZE, 3>::run(device, command_queue, results);
    small_general_sweep_size_2d<test_group_st_st_add<float>, SIZE, SIZE, 4>::run(device, command_queue, results);
    
    small_general_sweep_size_2d<test_group_st_st_add<half>,  SIZE, SIZE, 2>::run(device, command_queue, results);
    small_general_sweep_size_2d<test_group_st_st_add<half>,  SIZE, SIZE, 3>::run(device, command_queue, results);
    small_general_sweep_size_2d<test_group_st_st_add<half>,  SIZE, SIZE, 4>::run(device, command_queue, results);
    
    small_general_sweep_size_2d<test_group_st_st_add<bf16>,  SIZE, SIZE, 2>::run(device, command_queue, results);
    small_general_sweep_size_2d<test_group_st_st_add<bf16>,  SIZE, SIZE, 3>::run(device, command_queue, results);
    small_general_sweep_size_2d<test_group_st_st_add<bf16>,  SIZE, SIZE, 4>::run(device, command_queue, results);
    
    
    small_general_sweep_size_2d<test_group_st_t_add<float>, SIZE, SIZE, 2>::run(device, command_queue, results);
    small_general_sweep_size_2d<test_group_st_t_add<float>, SIZE, SIZE, 3>::run(device, command_queue, results);
    small_general_sweep_size_2d<test_group_st_t_add<float>, SIZE, SIZE, 4>::run(device, command_queue, results);
    
    small_general_sweep_size_2d<test_group_st_t_add<half>,  SIZE, SIZE, 2>::run(device, command_queue, results);
    small_general_sweep_size_2d<test_group_st_t_add<half>,  SIZE, SIZE, 3>::run(device, command_queue, results);
    small_general_sweep_size_2d<test_group_st_t_add<half>,  SIZE, SIZE, 4>::run(device, command_queue, results);
    
    small_general_sweep_size_2d<test_group_st_t_add<bf16>,  SIZE, SIZE, 2>::run(device, command_queue, results);
    small_general_sweep_size_2d<test_group_st_t_add<bf16>,  SIZE, SIZE, 3>::run(device, command_queue, results);
    small_general_sweep_size_2d<test_group_st_t_add<bf16>,  SIZE, SIZE, 4>::run(device, command_queue, results);
    
    
    small_general_sweep_size_2d<test_group_st_sv_row_add<float>, SIZE, SIZE, 2>::run(device, command_queue, results);
    small_general_sweep_size_2d<test_group_st_sv_row_add<float>, SIZE, SIZE, 3>::run(device, command_queue, results);
    small_general_sweep_size_2d<test_group_st_sv_row_add<float>, SIZE, SIZE, 4>::run(device, command_queue, results);
    
    small_general_sweep_size_2d<test_group_st_sv_row_add<half>,  SIZE, SIZE, 2>::run(device, command_queue, results);
    small_general_sweep_size_2d<test_group_st_sv_row_add<half>,  SIZE, SIZE, 3>::run(device, command_queue, results);
    small_general_sweep_size_2d<test_group_st_sv_row_add<half>,  SIZE, SIZE, 4>::run(device, command_queue, results);
    
    small_general_sweep_size_2d<test_group_st_sv_row_add<bf16>,  SIZE, SIZE, 2>::run(device, command_queue, results);
    small_general_sweep_size_2d<test_group_st_sv_row_add<bf16>,  SIZE, SIZE, 3>::run(device, command_queue, results);
    small_general_sweep_size_2d<test_group_st_sv_row_add<bf16>,  SIZE, SIZE, 4>::run(device, command_queue, results);

    
    small_general_sweep_size_2d<test_group_st_sv_col_add<float>, SIZE, SIZE, 2>::run(device, command_queue, results);
    small_general_sweep_size_2d<test_group_st_sv_col_add<float>, SIZE, SIZE, 3>::run(device, command_queue, results);
    small_general_sweep_size_2d<test_group_st_sv_col_add<float>, SIZE, SIZE, 4>::run(device, command_queue, results);
    
    small_general_sweep_size_2d<test_group_st_sv_col_add<half>,  SIZE, SIZE, 2>::run(device, command_queue, results);
    small_general_sweep_size_2d<test_group_st_sv_col_add<half>,  SIZE, SIZE, 3>::run(device, command_queue, results);
    small_general_sweep_size_2d<test_group_st_sv_col_add<half>,  SIZE, SIZE, 4>::run(device, command_queue, results);
    
    small_general_sweep_size_2d<test_group_st_sv_col_add<bf16>,  SIZE, SIZE, 2>::run(device, command_queue, results);
    small_general_sweep_size_2d<test_group_st_sv_col_add<bf16>,  SIZE, SIZE, 3>::run(device, command_queue, results);
    small_general_sweep_size_2d<test_group_st_sv_col_add<bf16>,  SIZE, SIZE, 4>::run(device, command_queue, results);
    //test_group_st_sv_col_add
}

////
//int main() {
////    should_write_outputs = 1;
//    NS::Error** error;
//    MTL::CopyAllDevices();
//    MTL::Device* device = MTL::CreateSystemDefaultDevice();
//    MTL::CommandQueue* command_queue = device->newCommandQueue();
//    test_data results;
//    group::shared::tile::maps::tests(device, command_queue, results);
//}


#endif

