#include "global_to_register.hpp"
#ifdef TEST_GROUP_MEMORY_VEC_GLOBAL_TO_REGISTER
#include "MetalSingle.hpp"

template<typename test, int S, int NUM_WORKERS, typename... args>
struct gtr_vec_wrapper_1d {
    using dtype = gmem_dtype<test>; // defaults to bf16 in global memory if the test doesn't specify.
    
    static void run(MTL::Device* device, MTL::CommandQueue* command_queue, test_data& results) {
        test_info this_result;
        this_result.label       = generate_test_name<false, S, NUM_WORKERS, args...>(test::test_identifier);
        this_result.kernel_name = generate_test_name<true, S, NUM_WORKERS, args...>(test::kernel_identifier);
        if constexpr (test::template valid<S, NUM_WORKERS, args...>::value) {
//            constexpr int B = 1, D = 1, R = 1, C = 1;
            constexpr int SIZE = S * 8 * NUM_WORKERS;
            MTL::Buffer* d_i;
            MTL::Buffer* d_o;
            std::vector<float> i_ref(SIZE);
            std::vector<float> o_ref(SIZE);
            initialize<dtype>(device, &d_i, &d_o, i_ref, o_ref);
            NS::String* kernel_name = NS::String::string(this_result.kernel_name.c_str(), NS::ASCIIStringEncoding);
            run_kernel<NUM_WORKERS>(device, command_queue, kernel_name, d_i, d_o);
            test::template host_func<S, NUM_WORKERS, args...>(i_ref, o_ref);
            this_result.result = validate<dtype>(d_i, d_o, i_ref, o_ref, this_result.label, S*8 * NUM_WORKERS);
            
        }
        else {
            this_result.result = test_result::INVALID;
        }
        results.push_back(this_result);
    }
};

template<typename T>
struct group_vec_gtr_load_store {
    using dtype = T;
    template<int L, int NW, typename... args> using valid = std::bool_constant<NW <= 4 && L <= 64>; // this is warp-level
    
    
    static inline const std::string test_identifier = std::is_same_v<T, bf16> ? "group_vec_gtr_loadstore=bf16" :
                                                      std::is_same_v<T, half> ? "group_vec_gtr_loadstore=half" :
                                                                                "group_vec_gtr_loadstore=float";
    static inline const std::string kernel_identifier = std::is_same_v<T, bf16> ? "group_vec_gtr_loadstore_bf16" :
                                                        std::is_same_v<T, half> ? "group_vec_gtr_loadstore_half" :
                                                                                  "group_vec_gtr_loadstore_float";
    template<int H, int NW, rv_layout::all L>
    static void host_func(const std::vector<float> &i_ref, std::vector<float> &o_ref) {
        o_ref = i_ref; // overwrite the whole thing
    }
};

template<typename test, int MAX_L=8, int NUM_WORKERS=1, typename... args>
using gtr_vec_sweep_size_1d = loop_s<gtr_vec_wrapper_1d, test, MAX_L, NUM_WORKERS, MAX_L, args...>;

void group::memory::vec::global_to_register::tests(MTL::Device* device, MTL::CommandQueue* command_queue, test_data &results) {
    std::cout << "\n ----- Starting ops/group/memory/vec/global_to_shared tests! -----\n" << std::endl;
    constexpr int SIZE = INTENSITY_1 ? 2  :
                         INTENSITY_2 ? 4  :
                         INTENSITY_3 ? 8  :
                         INTENSITY_4 ? 16 : -1;
    
    gtr_vec_sweep_size_1d<group_vec_gtr_load_store<float>, SIZE, 2, rv_layout::align>::run(device, command_queue, results);
    gtr_vec_sweep_size_1d<group_vec_gtr_load_store<float>, SIZE, 2, rv_layout::ortho>::run(device, command_queue, results);
    gtr_vec_sweep_size_1d<group_vec_gtr_load_store<float>, SIZE, 2, rv_layout::naive>::run(device, command_queue, results);
    
    gtr_vec_sweep_size_1d<group_vec_gtr_load_store<float>, SIZE, 3, rv_layout::align>::run(device, command_queue, results);
    gtr_vec_sweep_size_1d<group_vec_gtr_load_store<float>, SIZE, 3, rv_layout::ortho>::run(device, command_queue, results);
    gtr_vec_sweep_size_1d<group_vec_gtr_load_store<float>, SIZE, 3, rv_layout::naive>::run(device, command_queue, results);
    
    gtr_vec_sweep_size_1d<group_vec_gtr_load_store<float>, SIZE, 4, rv_layout::align>::run(device, command_queue, results);
    gtr_vec_sweep_size_1d<group_vec_gtr_load_store<float>, SIZE, 4, rv_layout::ortho>::run(device, command_queue, results);
    gtr_vec_sweep_size_1d<group_vec_gtr_load_store<float>, SIZE, 4, rv_layout::naive>::run(device, command_queue, results);

    gtr_vec_sweep_size_1d<group_vec_gtr_load_store<half>, SIZE, 2, rv_layout::align>::run(device, command_queue, results);
    gtr_vec_sweep_size_1d<group_vec_gtr_load_store<half>, SIZE, 2, rv_layout::ortho>::run(device, command_queue, results);
    gtr_vec_sweep_size_1d<group_vec_gtr_load_store<half>, SIZE, 2, rv_layout::naive>::run(device, command_queue, results);
    
    gtr_vec_sweep_size_1d<group_vec_gtr_load_store<half>, SIZE, 3, rv_layout::align>::run(device, command_queue, results);
    gtr_vec_sweep_size_1d<group_vec_gtr_load_store<half>, SIZE, 3, rv_layout::ortho>::run(device, command_queue, results);
    gtr_vec_sweep_size_1d<group_vec_gtr_load_store<half>, SIZE, 3, rv_layout::naive>::run(device, command_queue, results);
    
    gtr_vec_sweep_size_1d<group_vec_gtr_load_store<half>, SIZE, 4, rv_layout::align>::run(device, command_queue, results);
    gtr_vec_sweep_size_1d<group_vec_gtr_load_store<half>, SIZE, 4, rv_layout::ortho>::run(device, command_queue, results);
    gtr_vec_sweep_size_1d<group_vec_gtr_load_store<half>, SIZE, 4, rv_layout::naive>::run(device, command_queue, results);
    
    gtr_vec_sweep_size_1d<group_vec_gtr_load_store<bf16>, SIZE, 2, rv_layout::align>::run(device, command_queue, results);
    gtr_vec_sweep_size_1d<group_vec_gtr_load_store<bf16>, SIZE, 2, rv_layout::ortho>::run(device, command_queue, results);
    gtr_vec_sweep_size_1d<group_vec_gtr_load_store<bf16>, SIZE, 2, rv_layout::naive>::run(device, command_queue, results);
    
    gtr_vec_sweep_size_1d<group_vec_gtr_load_store<bf16>, SIZE, 3, rv_layout::align>::run(device, command_queue, results);
    gtr_vec_sweep_size_1d<group_vec_gtr_load_store<bf16>, SIZE, 3, rv_layout::ortho>::run(device, command_queue, results);
    gtr_vec_sweep_size_1d<group_vec_gtr_load_store<bf16>, SIZE, 3, rv_layout::naive>::run(device, command_queue, results);
    
    gtr_vec_sweep_size_1d<group_vec_gtr_load_store<bf16>, SIZE, 4, rv_layout::align>::run(device, command_queue, results);
    gtr_vec_sweep_size_1d<group_vec_gtr_load_store<bf16>, SIZE, 4, rv_layout::ortho>::run(device, command_queue, results);
    gtr_vec_sweep_size_1d<group_vec_gtr_load_store<bf16>, SIZE, 4, rv_layout::naive>::run(device, command_queue, results);
}

#endif

//
////
//int main() {
////    should_write_outputs = 1;
//    NS::Error** error;
//    MTL::CopyAllDevices();
//    MTL::Device* device = MTL::CreateSystemDefaultDevice();
//    MTL::CommandQueue* command_queue = device->newCommandQueue();
//    test_data results;
//    group::memory::vec::global_to_register::tests(device, command_queue, results);
//}
