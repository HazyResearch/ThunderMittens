// testing_utils.hpp
#pragma once
#include <cassert>
#include <vector>
#include <iostream>
#include <fstream>
#include <random>
#include <algorithm>
#include <cmath>
#include <type_traits>
#include "testing_types.hpp"
#include "MetalSingle.hpp"

/* --------------------  TEST STRUCTS  -------------------- */

enum test_result {
    PASSED = 0,
    FAILED = 1,
    INVALID = 2 // This is a useful one for tests that are only defined for certain template specializations, but we still want to sweep.
};
struct test_info {
    std::string label;
    std::string kernel_name;
    test_result result;
};
using test_data = std::vector<test_info>;


/* --------------------  TEMPLATE METAPROGRAMMING UTILS  -------------------- */

// 1D wrapper
template<template<typename,int,int,typename...> typename base, typename test, int MAX_S, int NUM_WORKERS, int S, typename... args>
struct loop_s {
    static void run(MTL::Device* device, MTL::CommandQueue* command_queue, test_data& results) {
        if constexpr (S > 1) {
            loop_s<base, test, MAX_S, NUM_WORKERS, S-1, args...>::run(device, command_queue, results);
        }
        base<test, S, NUM_WORKERS, args...>::run(device, command_queue, results);
    }
};

// 2D wrappers
template<template<typename,int,int,int,typename...> typename base, typename test, int MAX_H, int MAX_W, int NUM_WORKERS, int H, int W, typename... args>
struct loop_w {
    static void run(MTL::Device* device, MTL::CommandQueue* command_queue, test_data& results) {
        if constexpr (W > 1) {
            loop_w<base, test, MAX_H, MAX_W, NUM_WORKERS, H, W-1, args...>::run(device, command_queue, results);
        }
        base<test, H, W, NUM_WORKERS, args...>::run(device, command_queue, results);
    }
};
template<template<typename,int,int,int,typename...> typename base, typename test, int MAX_H, int MAX_W, int NUM_WORKERS, int H, typename... args>
struct loop_h {
    static void run(MTL::Device* device, MTL::CommandQueue* command_queue, test_data& results) {
        if constexpr (H > 1) {
            loop_h<base, test, MAX_H, MAX_W, NUM_WORKERS, H-1, args...>::run(device, command_queue, results);
        }
        loop_w<base, test, MAX_H, MAX_W, NUM_WORKERS, H, MAX_W, args...>::run(device, command_queue, results);
    }
};



/* --------------------  TEST INITIALIZE+VALIDATE FUNCS  -------------------- */

enum initializers {
    RANDOM = 0, // uniform random init. useful for confirming correctness.
    ARANGE = 1, // write an increasing sequence into i_ref and d_i arrays. useful for debugging memory movement.
    NONE   = 2  // use whatever values were already in i_ref. useful for detailed debugging.
};
template<typename T, initializers initializer=initializers::RANDOM, int SEED=42>
void initialize(MTL::Device* device,
                MTL::Buffer **d_i, MTL::Buffer **d_o,
                std::vector<float> &i_ref, std::vector<float> &o_ref) {
    const int input_size  = i_ref.size();
    const int output_size = o_ref.size();
    
    // Initialize matrices
    std::vector<T> i_t(input_size);
    
    std::mt19937 gen(SEED); // Standard mersenne_twister_engine
    std::uniform_real_distribution<float> dis(-1.0, 1.0);
    for(int idx = 0; idx < input_size; idx++) {
        float f;
        if constexpr (initializer == initializers::RANDOM) {
            f = dis(gen);
        }
        else if constexpr (initializer == initializers::ARANGE) {
            f = float(idx);
        }
        else {
            f = i_ref[idx];
        }
        if constexpr (std::is_same_v<T, bf16>) {
            i_t[idx] = float_to_bf16(f); // fill in for transfer to device
            i_ref[idx] = bf16_to_float(i_t[idx]); // ensure lossiness of fp16 is captured on cpu
        }
        else if constexpr (std::is_same_v<T, float>) {
            i_t[idx] = f;
            i_ref[idx] = f;
        }
        else if constexpr (std::is_same_v<T, half>) {
            i_t[idx] = float_to_half(f);
            i_ref[idx] = half_to_float(i_t[idx]);
        }
        else {
            assert(false && "Unsupported data type");
        }
    }
    uintptr_t input_len = input_size * sizeof(T);
    uintptr_t output_len = input_size * sizeof(T);
    *d_i = device->newBuffer(input_len, MTL::ResourceStorageModeShared);
    *d_o = device->newBuffer(output_len, MTL::ResourceStorageModeShared);
    memcpy((*d_i)->contents(), i_t.data(), input_len);
}
extern int should_write_outputs;
template<typename T>
test_result validate(MTL::Buffer *d_i, MTL::Buffer *d_o,
                     const std::vector<float> &i_ref, std::vector<float> &o_ref,
                     std::string test_name,
                     int cols, float eps=5e-2) { // default eps has to be fairly high due to lots of different types
    const int input_size  = i_ref.size();
    const int output_size = o_ref.size();
    T* o_t = (T*)d_o->contents();
    float *o = new float[output_size];
    for(int idx = 0; idx < output_size; idx++) {
        if constexpr (std::is_same_v<T, bf16>) {
            o[idx]     = bf16_to_float(o_t[idx]);
            o_ref[idx] = bf16_to_float(float_to_bf16(o_ref[idx]));
        }
        else if constexpr (std::is_same_v<T, half>) {
            o[idx]     = half_to_float(o_t[idx]);
            o_ref[idx] = half_to_float(float_to_half(o_ref[idx]));
        }
        else if constexpr(std::is_same_v<T, float>) {
            o[idx] = o_t[idx];
            o_ref[idx] = o_ref[idx];
        }
        else {
            assert(false && "Unsupported data type");
        }
    }
    std::cout << "test `" << test_name << "`";

//    printf("\nRef:\n");
//    for(int i = 0; i < output_size; i++) {
//        printf("%.3f, ", o_ref[i]);
//        if (i % cols == cols - 1) {
//            printf("\n");
//        }
//    }
//    printf("\nCus:\n");
//    for(int i = 0; i < output_size; i++) {
//        printf("%.3f, ", o[i]);
//        if (i % cols == cols - 1) {
//            printf("\n");
//        }
//    }
//    printf("\n");
    
    bool good = true;
    for(int i = 0; i < output_size; i++) {
        if(fabs(o_ref[i] - o[i]) > eps) {
            good = false;
            break;
        }
    }
    
    if(good) std::cout << " -- PASSED" << std::endl;
    else std::cout << " ----- ALERT! FAILED test `" << test_name << "` -----" << std::endl;
    if(should_write_outputs && !good) {
        
        std::ofstream reffile("/../outputs/"+test_name+"_ref.txt");
        std::ofstream outfile("/../outputs/"+test_name+"_out.txt");
        for(int i = 0; i < output_size; i++) {
            reffile << o_ref[i] << ' ';
            outfile << o[i] << ' ';
            if(i%cols == cols-1) {
                reffile << '\n';
                outfile << '\n';
            }
        }
        reffile << "\n\n\nINPUTS:\n\n";
        for(int i = 0; i < input_size; i++) {
            reffile << i_ref[i] << ' ';
            if(i%cols == cols-1) {
                reffile << '\n';
            }
        }
        reffile.close();
        outfile.close();
    }
    d_i->release();
    d_o->release();
    delete[] o;
    return good ? test_result::PASSED : test_result::FAILED;
}

