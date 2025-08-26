#pragma once

/* testing_commons.cuh
 *
 * This file contains a bunch of moderately test-specific utils.
 * For example, test_name constructors and __device__ kernel wrappers.
 * This file is distinguished from testing_utils.cuh in that you
 * might need to add to this file in order to add more tests,
 * but you shouldn't need to modify that testing_utils at all.
 */

// testing_commons.hpp
#include "testing_utils.hpp"

namespace rt_layout {
struct row {};
struct col {};

template<typename T>
concept all = std::is_same_v<T, row> || std::is_same_v<T, col>;
}

namespace rv_layout {
struct naive {};
struct ortho {};
struct align {};

template<typename T>
concept all = std::is_same_v<T, naive> || std::is_same_v<T, ortho> || std::is_same_v<T, align>;
}

template<typename T>
concept T1 = std::is_same_v<T, float>  || std::is_same_v<T, bf16  > || std::is_same_v<T, half>; // could add half_2 later if implemented.

/* ---------- TEST NAMES ---------- */

// This how we generate parameterized names for tests.
// test_id is defined by the test, like "reg_mma" --
// then these templates build the rest of the test name.
// Note use of concepts to prevent template arg collisions!

// 1D test names
template<bool KERNEL, int S, int NW> std::string generate_test_name(std::string test_id) {
    
    if constexpr (KERNEL) {
        std::string label = test_id+"_"+std::to_string(S);
        if constexpr (NW > 1) {
            label += "_"+std::to_string(NW)+"warps";
        }
        return label;
    }
    std::string label = test_id+"_"+std::to_string(S);
    if constexpr (NW > 1) {
        label += "_["+std::to_string(NW)+"warps]";
    }
    
    return label;
}
template<bool KERNEL, int S, int NW, rt_layout::all L> std::string generate_test_name(std::string test_id) {
    std::string label = generate_test_name<KERNEL,S,NW>(test_id);
    if constexpr (KERNEL) {
        if constexpr (std::is_same_v<L, rt_layout::row>) label += "_rt_row_layout";
        else label += "_rt_col_layout";
        return label;
    }
    if constexpr (std::is_same_v<L, rt_layout::row>) label += "_[rt_row_layout]";
    else label += "_[rt_col_layout]";
    
    return label;
}
template<bool KERNEL, int S, int NW, rt_layout::all L1, rt_layout::all L2> std::string generate_test_name(std::string test_id) {
    std::string label = generate_test_name<KERNEL,S,NW,L1>(test_id);
    if constexpr (KERNEL) {
        if constexpr (std::is_same_v<L2, rt_layout::row>) label += "_rt_row_layout";
        else label += "_rt_col_layout";
        return label;
    }
    if constexpr (std::is_same_v<L2, rt_layout::row>) label += "_[rt_row_layout]";
    else label += "_[rt_col_layout]";
    
    return label;
}
template<bool KERNEL, int S, int NW, rv_layout::all L> std::string generate_test_name(std::string test_id) {
    std::string label = generate_test_name<KERNEL,S,NW>(test_id);
    if constexpr (KERNEL) {
        if constexpr (std::is_same_v<L, rv_layout::naive>) label += "_rv_naive_layout";
        else if constexpr (std::is_same_v<L, rv_layout::ortho>) label += "_rv_ortho_layout";
        else label += "_rv_align_layout";
        return label;
    }
    if constexpr (std::is_same_v<L, rv_layout::naive>) label += "_[rv_naive_layout]";
    else if constexpr (std::is_same_v<L, rv_layout::ortho>) label += "_[rv_ortho_layout]";
    else label += "_[rv_align_layout]";
    
    return label;
}
template<bool KERNEL, int S, int NW, rv_layout::all L1, rv_layout::all L2> std::string generate_test_name(std::string test_id) {
    std::string label = generate_test_name<KERNEL,S,NW,L1>(test_id);
    if constexpr (KERNEL) {
        if constexpr (std::is_same_v<L2, rv_layout::naive>) label += "_rv_naive_layout";
        else if constexpr (std::is_same_v<L2, rv_layout::ortho>) label += "_rv_ortho_layout";
        else label += "_rv_align_layout";
        return label;
    }
    if constexpr (std::is_same_v<L2, rv_layout::naive>) label += "_[rv_naive_layout]";
    else if constexpr (std::is_same_v<L2, rv_layout::ortho>) label += "_[rv_ortho_layout]";
    else label += "_[rv_align_layout]";

    return label;
}

// 2D test names

template<bool KERNEL, int H, int W, int NW> std::string generate_test_name(std::string test_id) {
    
    if constexpr (KERNEL) {
        std::string label = test_id+"_"+std::to_string(H)+"x"+std::to_string(W);
        if constexpr (NW > 1) {
            label += "_"+std::to_string(NW)+"warps";
        }
        return label;
    }
    std::string label = test_id+"_["+std::to_string(H)+"x"+std::to_string(W)+"]";
    if constexpr (NW > 1) {
        label += "_["+std::to_string(NW)+"warps]";
    }
    
    return label;
}
template <typename T> concept integral_wrapper = std::is_integral_v<decltype(T::value)>;

template<bool KERNEL, int H, int W, int NW, integral_wrapper _K> std::string generate_test_name(std::string test_id) {
    constexpr int K = _K::value;
    if constexpr (KERNEL) {
        std::string label = test_id+"_"+std::to_string(H)+"x"+std::to_string(K)+"x"+std::to_string(W)+"";
        if constexpr (NW > 1) {
            label += "_"+std::to_string(NW)+"warps";
        }
        return label;
    }
    std::string label = test_id+"_["+std::to_string(H)+"x"+std::to_string(K)+"x"+std::to_string(W)+"]";
    if constexpr (NW > 1) {
        label += "_["+std::to_string(NW)+"warps]";
    }
    return label;
}
template<bool KERNEL, int H, int W, int NW, rt_layout::all L> std::string generate_test_name(std::string test_id) {
    std::string label = generate_test_name<KERNEL,H,W,NW>(test_id);
    if constexpr (KERNEL) {
        if constexpr (std::is_same_v<L, rt_layout::row>) label += "_rt_row_layout";
        else label += "_rt_col_layout";
        return label;
    }
        if constexpr (std::is_same_v<L, rt_layout::row>) label += "_[rt_row_layout]";
        else label += "_[rt_col_layout]";
    
    return label;
}
template<bool KERNEL, int H, int W, int NW, integral_wrapper _J, integral_wrapper _K> std::string generate_test_name(std::string test_id) {
    constexpr int J = _J::value, K = _K::value;
    if constexpr (KERNEL) {
        std::string label = test_id+"_"+std::to_string(H)+"x"+std::to_string(W)+"_"+std::to_string(J)+"x"+std::to_string(K);
        if constexpr (NW > 1) {
            label += "_"+std::to_string(NW)+"warps";
        }
        return label;
    }
    std::string label = test_id+"_["+std::to_string(H)+"x"+std::to_string(W)+"_"+std::to_string(J)+"x"+std::to_string(K)+"]";
    if constexpr (NW > 1) {
        label += "_["+std::to_string(NW)+"warps]";
    }
    
    return label;
}
template<bool KERNEL, int H, int W, int NW, T1 T2, T1 U2> std::string generate_test_name(std::string test_id) {
    std::string label = generate_test_name<KERNEL,H,W,NW>(test_id);
    if constexpr (KERNEL) {
        if constexpr (std::is_same_v<U2, float>) label += "_float_to";
        else if constexpr (std::is_same_v<U2, bf16>) label += "_bf16_to";
        else label += "_half_to";
        if constexpr (std::is_same_v<T2, float>) label += "float";
        else if constexpr (std::is_same_v<T2, bf16>) label += "bf16";
        else label += "half";
        return label;
    }
        if constexpr (std::is_same_v<U2, float>) label += "_[float->";
        else if constexpr (std::is_same_v<U2, bf16>) label += "_[bf16->";
        else label += "_[half->";
        if constexpr (std::is_same_v<T2, float>) label += "float]";
        else if constexpr (std::is_same_v<T2, bf16>) label += "bf16]";
        else label += "half]";
    
    return label;
}


/* ---------- TEST WRAPPERS ---------- */

// These are wrappers to make it really easy to call and run tests.
// The basic wrappers:
// - Check if the test is valid and not compile it otherwise (the if constexpr)
// - Initialize input and output memory on both host and device
// - Call test functions on host and device
// - Validate outputs, append the result to test_data& results
// - Cleanup
// Additionally, the templated wrappers:
// - Loop through lots of template args in a grid to check validity.

template<typename T> concept has_dtype = requires { typename T::dtype; };
template<typename T>  struct gmem_wrapper    { using dtype = bf16; };
template<has_dtype T> struct gmem_wrapper<T> { using dtype = typename T::dtype; };
template<typename T> using gmem_dtype = typename gmem_wrapper<T>::dtype;

// ----- 1D Wrappers -----
template<int NW>
MTL::CommandBuffer* run_kernel_async(MTL::Device* device, MTL::CommandQueue* command_queue, NS::String* kernel_name, MTL::Buffer* input, MTL::Buffer* output) {
    NS::Error** error;
    MTL::Size threadgroup_size(32, NW, 1);
    MTL::Size num_threadgroups(1, 1, 1);
    MTL::CommandBuffer* command_buffer = command_queue->commandBuffer();
    MTL::ComputeCommandEncoder* compute_encoder = command_buffer->computeCommandEncoder();
    MTL::Library* default_library = device->newDefaultLibrary();
    MTL::Function* kernel = default_library->newFunction(kernel_name);
    MTL::ComputePipelineState* kernelPSO = device->newComputePipelineState(kernel, error);
    compute_encoder->setComputePipelineState(kernelPSO);
    compute_encoder->setBuffer(input, 0, 0);
    compute_encoder->setBuffer(output, 0, 1);
    compute_encoder->dispatchThreadgroups(num_threadgroups, threadgroup_size);
    compute_encoder->endEncoding();
    command_buffer->commit();
    return command_buffer;
}


template<int NW>
void run_kernel(MTL::Device* device, MTL::CommandQueue* command_queue, NS::String* kernel_name, MTL::Buffer* input, MTL::Buffer* output) {
    MTL::CommandBuffer* command_buffer = run_kernel_async<NW>(device, command_queue, kernel_name, input, output);
    command_buffer->waitUntilCompleted();
}

template<typename test, int S, int NUM_WORKERS, typename... args>
struct warp_wrapper_1d {
    using dtype = gmem_dtype<test>; // defaults to bf16 in global memory if the test doesn't specify.
    
    static void run(MTL::Device* device, MTL::CommandQueue* command_queue, test_data& results) {
        test_info this_result;
        this_result.label       = generate_test_name<false, S, NUM_WORKERS, args...>(test::test_identifier);
        this_result.kernel_name = generate_test_name<true, S, NUM_WORKERS, args...>(test::kernel_identifier);
        if constexpr (test::template valid<S, NUM_WORKERS, args...>::value) {
//            constexpr int B = 1, D = 1, R = 1, C = 1;
            constexpr int SIZE = S * 8;
            MTL::Buffer* d_i;
            MTL::Buffer* d_o;
            std::vector<float> i_ref(SIZE);
            std::vector<float> o_ref(SIZE);
            initialize<dtype>(device, &d_i, &d_o, i_ref, o_ref);
            NS::String* kernel_name = NS::String::string(this_result.kernel_name.c_str(), NS::ASCIIStringEncoding);
//            std::cout << this_result.kernel_name << std::endl;
            run_kernel<NUM_WORKERS>(device, command_queue, kernel_name, d_i, d_o);
            test::template host_func<S, NUM_WORKERS, args...>(i_ref, o_ref);
            this_result.result = validate<dtype>(d_i, d_o, i_ref, o_ref, this_result.label, S*8);
            
        }
        else {
            this_result.result = test_result::INVALID;
        }
        results.push_back(this_result);
    }
};

template<typename test, int H, int W, int NUM_WORKERS, typename... args>
struct warp_wrapper_2d {
    using dtype = gmem_dtype<test>; // defxrxrxbaults to bf16 in global memory if the test doesn't specify.
    
    static void run(MTL::Device* device, MTL::CommandQueue* command_queue, test_data& results) {
        test_info this_result;
        this_result.label       = generate_test_name<false, H,W,NUM_WORKERS, args...>(test::test_identifier);
        this_result.kernel_name = generate_test_name<true, H,W,NUM_WORKERS, args...>(test::kernel_identifier);
        
        if constexpr (test::template valid<H, W, NUM_WORKERS, args...>::value) {
            constexpr int B = 3, D = 1, R = 4, C = 5;
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

template<typename test, int H, int W, int NUM_WORKERS, typename... args>
struct small_warp_wrapper_2d {
    using dtype = gmem_dtype<test>; // defaults to bf16 in global memory if the test doesn't specify.
    
    static void run(MTL::Device* device, MTL::CommandQueue* command_queue, test_data& results) {
        test_info this_result;
        this_result.label       = generate_test_name<false, H,W,NUM_WORKERS, args...>(test::test_identifier);
        this_result.kernel_name = generate_test_name<true, H,W,NUM_WORKERS, args...>(test::kernel_identifier);
        if constexpr (test::template valid<H, W, NUM_WORKERS, args...>::value) {
            constexpr int B = 1, D = 1, R = 1, C = 1;
            constexpr int SIZE = H*W*8*8 * B * D * R * C;
            MTL::Buffer* d_i;
            MTL::Buffer* d_o;
            std::vector<float> i_ref(SIZE);
            std::vector<float> o_ref(SIZE);
            initialize<dtype>(device, &d_i, &d_o, i_ref, o_ref);
            NS::String* kernel_name = NS::String::string(this_result.kernel_name.c_str(), NS::ASCIIStringEncoding);
            run_kernel<NUM_WORKERS>(device, command_queue, kernel_name, d_i, d_o);
            test::template host_func<H, W, NUM_WORKERS, args...>(i_ref, o_ref);
            this_result.result = validate<dtype>(d_i, d_o, i_ref, o_ref, this_result.label, W*8);
            
        }
        else {
            this_result.result = test_result::INVALID;
        }
        results.push_back(this_result);
    }
};


template<typename test, int MAX_L=8, int NUM_WORKERS=1, typename... args>
using general_sweep_size_1d = loop_s<warp_wrapper_1d, test, MAX_L, NUM_WORKERS, MAX_L, args...>;
template<typename test, int MAX_L=8, typename... args>
using general_sweep_size_1d_warp = general_sweep_size_1d<test, MAX_L, 1, args...>;



template<typename test, int MAX_H=8, int MAX_W=8, int NUM_WORKERS=1, typename... args>
using general_sweep_size_2d = loop_h<warp_wrapper_2d, test, MAX_H, MAX_W, NUM_WORKERS, MAX_H, args...>;
template<typename test, int MAX_H=8, int MAX_W=8, typename... args>
using general_sweep_size_2d_warp = general_sweep_size_2d<test, MAX_H, MAX_W, 1, args...>;

template<typename test, int MAX_H=8, int MAX_W=8, int NW=4, typename... args>
using general_sweep_size_NW_2d_warp = general_sweep_size_2d<test, MAX_H, MAX_W, NW, args...>;

template<typename test, int MAX_H=8, int MAX_W=8, int NUM_WORKERS=1, typename... args>
using small_general_sweep_size_2d = loop_h<small_warp_wrapper_2d, test, MAX_H, MAX_W, NUM_WORKERS, MAX_H, args...>;
template<typename test, int MAX_H=8, int MAX_W=8, typename... args>
using small_general_sweep_size_2d_warp = small_general_sweep_size_2d<test, MAX_H, MAX_W, 1, args...>;
