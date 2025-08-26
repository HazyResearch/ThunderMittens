// testing_types.hpp
#pragma once
#include <cstdint>

typedef __fp16 half;       // Define half type using __fp16
typedef __bf16 bf16;   // Define bfloat16 type using __bf16

// Utility function to convert __fp16 to uint16_t using pointer casting
static inline  uint16_t fp16_to_uint16(half h) {
    return *reinterpret_cast<uint16_t*>(&h);
}

// Utility function to convert uint16_t to __fp16 using pointer casting
static inline half uint16_to_fp16(uint16_t u) {
    return *reinterpret_cast<half*>(&u);
}

// Utility function to convert __bf16 to uint16_t using pointer casting
static inline uint16_t bf16_to_uint16(bf16 bf) {
    return *reinterpret_cast<uint16_t*>(&bf);
}

// Utility function to convert uint16_t to __bf16 using pointer casting
static inline bf16 uint16_to_bf16(uint16_t u) {
    return *reinterpret_cast<bf16*>(&u);
}

// Convert float to bfloat16
static inline bf16 float_to_bf16(float f) {
    uint32_t* f_bits = reinterpret_cast<uint32_t*>(&f);
    uint16_t bf16 = static_cast<uint16_t>(*f_bits >> 16); // Truncate lower 16 bits
    return uint16_to_bf16(bf16);
}

// Convert bfloat16 to float
static inline float bf16_to_float(bf16 bf16) {
    uint16_t bf16_bits = bf16_to_uint16(bf16);
    uint32_t temp = static_cast<uint32_t>(bf16_bits) << 16;
    return *reinterpret_cast<float*>(&temp);
}

// Convert float to half
static inline half float_to_half(float f) {
    uint32_t* f_bits = reinterpret_cast<uint32_t*>(&f);
    
    // Extract sign, exponent, and mantissa
    uint16_t sign = (*f_bits >> 16) & 0x8000;
    int16_t exponent = ((*f_bits >> 23) & 0xFF) - 112;  // Adjust exponent bias
    uint16_t mantissa = (*f_bits >> 13) & 0x03FF;       // Truncate mantissa
    
    // Handle special cases
    if (exponent <= 0) {
        // Underflow to zero
        return uint16_to_fp16(sign);
    } else if (exponent >= 31) {
        // Overflow to infinity
        return uint16_to_fp16(sign | 0x7C00);
    }
    
    return uint16_to_fp16(sign | (exponent << 10) | mantissa);
}

// Convert half to float
static inline float half_to_float(half h) {
    uint16_t h_bits = fp16_to_uint16(h);
    uint32_t sign = (h_bits & 0x8000) << 16;
    int32_t exponent = (h_bits >> 10) & 0x1F;
    uint32_t mantissa = h_bits & 0x03FF;
    
    if (exponent == 0) {
        // Subnormal number or zero
        if (mantissa == 0) {
            return *reinterpret_cast<float*>(&sign);  // Zero
        } else {
            // Denormalized number
            exponent = 1;
        }
    } else if (exponent == 31) {
        // Infinity or NaN
        uint32_t inf_or_nan = sign | 0x7F800000 | (mantissa << 13);
        return *reinterpret_cast<float*>(&inf_or_nan);
    }
    
    exponent += 112;  // Adjust exponent bias
    
    uint32_t result_bits = sign | (exponent << 23) | (mantissa << 13);
    return *reinterpret_cast<float*>(&result_bits);
}

// Convert bfloat16 to half
static inline half bf16_to_half(bf16 bf16) {
    float temp = bf16_to_float(bf16);  // Convert to float first
    return float_to_half(temp);            // Convert float to half
}

// Convert half to bfloat16
static inline bf16 half_to_bf16(half h) {
    float temp = half_to_float(h);         // Convert to float first
    return float_to_bf16(temp);        // Convert float to bfloat16
}

