#include "shared_to_register.hpp"
#ifdef TEST_GROUP_MEMORY_TILE_SHARED_TO_REGISTER
#include "MetalSingle.hpp"

template<typename test, int H, int W, int NUM_WORKERS, typename... args>
struct warp_wrapper_2d_str {
    using dtype = gmem_dtype<test>;
    
    static void run(MTL::Device* device, MTL::CommandQueue* command_queue, test_data& results) {
        test_info this_result;
        this_result.label       = generate_test_name<false, H,W,NUM_WORKERS, args...>(test::test_identifier);
        this_result.kernel_name = generate_test_name<true, H,W,NUM_WORKERS, args...>(test::kernel_identifier);
        
        if constexpr (test::template valid<H, W, NUM_WORKERS, args...>::value) {
            constexpr int B = 3, D = 1, R = NUM_WORKERS, C = 5;
            constexpr int SIZE = H*W*8*8 * B * D * R * C;
            MTL::Buffer* d_i;
            MTL::Buffer* d_o;
            std::vector<float> i_ref(SIZE);
            std::vector<float> o_ref(SIZE);
            initialize<dtype>(device, &d_i, &d_o, i_ref, o_ref);
            NS::String* kernel_name = NS::String::string(this_result.kernel_name.c_str(), NS::ASCIIStringEncoding);
            run_kernel<NUM_WORKERS>(device, command_queue, kernel_name, d_i, d_o);
            test::template host_func<H, W, NUM_WORKERS, args...>(i_ref, o_ref);
            this_result.result = validate<dtype>(d_i, d_o, i_ref, o_ref, this_result.label, C*W*8);
            
        }
        else {
            this_result.result = test_result::INVALID;
        }
        results.push_back(this_result);
    }
};



template<typename T>
struct test_group_str_load_store {
    using dtype = T;
    template<int H, int W, int NW, typename... args> using valid = std::bool_constant<NW <= 4 && (H*NW*8)*(W*8)*2*sizeof(T) <= 32768>; // this is warp-level
    
    
    static inline const std::string test_identifier = std::is_same_v<T, bf16> ? "group_str_loadstore=bf16" :
                                                      std::is_same_v<T, half> ? "group_str_loadstore=half" :
                                                                                "group_str_loadstore=float";
    static inline const std::string kernel_identifier = std::is_same_v<T, bf16> ? "group_str_loadstore_bf16" :
                                                        std::is_same_v<T, half> ? "group_str_loadstore_half" :
                                                                                  "group_str_loadstore_float";
    template<int H, int W, int NW, rt_layout::all L>
    static void host_func(const std::vector<float> &i_ref, std::vector<float> &o_ref) {
        o_ref = i_ref; // overwrite the whole thing
    }
};

template<typename test, int MAX_H=8, int MAX_W=8, int NUM_WORKERS=4, typename... args>
using str_sweep_size_2d_group = loop_h<warp_wrapper_2d_str, test, MAX_H, MAX_W, NUM_WORKERS, MAX_H, args...>;

void group::memory::tile::shared_to_register::tests(MTL::Device* device, MTL::CommandQueue* command_queue, test_data &results) {
    std::cout << "\n ----- Starting ops/group/memory/tile/shared_to_register tests! -----\n" << std::endl;
    constexpr int SIZE = INTENSITY_1 ? 2  :
                         INTENSITY_2 ? 4  :
                         INTENSITY_3 ? 8  :
                         INTENSITY_4 ? 16 : -1;
    str_sweep_size_2d_group<test_group_str_load_store<float>, SIZE, SIZE, 2, rt_layout::row>::run(device, command_queue, results);
    str_sweep_size_2d_group<test_group_str_load_store<float>, SIZE, SIZE, 3, rt_layout::row>::run(device, command_queue, results);
    str_sweep_size_2d_group<test_group_str_load_store<float>, SIZE, SIZE, 4, rt_layout::row>::run(device, command_queue, results);
    str_sweep_size_2d_group<test_group_str_load_store<float>, SIZE, SIZE, 2, rt_layout::col>::run(device, command_queue, results);
    str_sweep_size_2d_group<test_group_str_load_store<float>, SIZE, SIZE, 3, rt_layout::col>::run(device, command_queue, results);
    str_sweep_size_2d_group<test_group_str_load_store<float>, SIZE, SIZE, 4, rt_layout::col>::run(device, command_queue, results);
    
    str_sweep_size_2d_group<test_group_str_load_store<half>, SIZE, SIZE, 2, rt_layout::row>::run(device, command_queue, results);
    str_sweep_size_2d_group<test_group_str_load_store<half>, SIZE, SIZE, 3, rt_layout::row>::run(device, command_queue, results);
    str_sweep_size_2d_group<test_group_str_load_store<half>, SIZE, SIZE, 4, rt_layout::row>::run(device, command_queue, results);
    str_sweep_size_2d_group<test_group_str_load_store<half>, SIZE, SIZE, 2, rt_layout::col>::run(device, command_queue, results);
    str_sweep_size_2d_group<test_group_str_load_store<half>, SIZE, SIZE, 3, rt_layout::col>::run(device, command_queue, results);
    str_sweep_size_2d_group<test_group_str_load_store<half>, SIZE, SIZE, 4, rt_layout::col>::run(device, command_queue, results);
    
    str_sweep_size_2d_group<test_group_str_load_store<bf16>, SIZE, SIZE, 2, rt_layout::row>::run(device, command_queue, results);
    str_sweep_size_2d_group<test_group_str_load_store<bf16>, SIZE, SIZE, 3, rt_layout::row>::run(device, command_queue, results);
    str_sweep_size_2d_group<test_group_str_load_store<bf16>, SIZE, SIZE, 4, rt_layout::row>::run(device, command_queue, results);
    str_sweep_size_2d_group<test_group_str_load_store<bf16>, SIZE, SIZE, 2, rt_layout::col>::run(device, command_queue, results);
    str_sweep_size_2d_group<test_group_str_load_store<bf16>, SIZE, SIZE, 3, rt_layout::col>::run(device, command_queue, results);
    str_sweep_size_2d_group<test_group_str_load_store<bf16>, SIZE, SIZE, 4, rt_layout::col>::run(device, command_queue, results);
    
}

#endif



//int main() {
////    should_write_outputs = 1;
//    
//    NS::Error** error;
//    MTL::CopyAllDevices();
//    MTL::Device* device = MTL::CreateSystemDefaultDevice();
//    std::cout << device->maxThreadgroupMemoryLength() << std::endl;
//    MTL::CommandQueue* command_queue = device->newCommandQueue();
//    test_data results;
//    group::memory::tile::shared_to_register::tests(device, command_queue, results);
//    
//}
