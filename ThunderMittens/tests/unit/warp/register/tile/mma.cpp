#include "mma.hpp"

#ifdef TEST_WARP_REGISTER_TILE_MMA


template<typename test, int H, int W, int NUM_WORKERS, typename _K, typename... args>
struct mma_wrapper_2d {
    using dtype = float; // defaults to bf16 in global memory if the test doesn't specify.
    static void run(MTL::Device* device, MTL::CommandQueue* command_queue, test_data& results) {
        constexpr int K = _K::value;
        test_info this_result;
        this_result.label       = generate_test_name<false, H, W, NUM_WORKERS, _K, args...>(test::test_identifier);
        this_result.kernel_name = generate_test_name<true,  H, W, NUM_WORKERS, _K, args...>(test::kernel_identifier);
        if constexpr (test::template valid<H, W, NUM_WORKERS, _K, args...>::value) {
            MTL::Buffer* d_i;
            MTL::Buffer* d_o;
            std::vector<float> i_ref((H+W)*K*64);
            std::vector<float> o_ref(H*W*64);
//            initialize<dtype, initializers::ARANGE>(device, &d_i, &d_o, i_ref, o_ref);
            initialize<dtype>(device, &d_i, &d_o, i_ref, o_ref);
            NS::String* kernel_name = NS::String::string(this_result.kernel_name.c_str(), NS::ASCIIStringEncoding);
            std::cout << this_result.kernel_name << std::endl;
            run_kernel<NUM_WORKERS>(device, command_queue, kernel_name, d_i, d_o);
            
            test::template host_func<H, W, NUM_WORKERS, _K, args...>(i_ref, o_ref);
            this_result.result = validate<dtype>(d_i, d_o, i_ref, o_ref, this_result.label, W * 8);
        }
        else {
            this_result.result = test_result::INVALID;
        }
        results.push_back(this_result);
    }
};


template <typename T>
struct test_mma_AB {
    template<int H, int W, int NW, typename K> using valid = std::bool_constant<NW == 1 && (2*W*H+W*K::value+H*K::value)<=64>; // this is warp-level
    static inline const std::string test_identifier = "reg_mma_AB";
    static inline const std::string kernel_identifier = "reg_mma_AB";
    using dtype = float;
    
    template<int _N, int _M, int NW, typename _K>
    static void host_func(const std::vector<float> &i_ref, std::vector<float> &o_ref) {
        constexpr int K = _K::value * 8;
        constexpr int N = _N * 8;
        constexpr int M = _M * 8;
        
        for(int y = 0; y < N; y++) {
            for(int x = 0; x < M; x++) {
                float sum = 0;
                for(int k = 0; k < K; k++) {
                    sum += i_ref[y*K + k]*i_ref[(N*K) + k*M + x];
                }
                o_ref[y*M + x] = sum;
            }
        }
    }
};

template <typename T>
struct test_mma_ABt {
    template<int H, int W, int NW, typename K> using valid = std::bool_constant<NW == 1 && (2*W*H+W*K::value+H*K::value)<=64>; // this is warp-level
    static inline const std::string test_identifier = "reg_mma_ABt";
    static inline const std::string kernel_identifier = "reg_mma_ABt";
    using dtype = float;
    /*
     A: NxK
     B: MxK
     C: NxM
     */
    template<int _N, int _M, int NW, typename _K>
    static void host_func(const std::vector<float> &i_ref, std::vector<float> &o_ref) {
        constexpr int K = _K::value * 8;
        constexpr int N = _N * 8;
        constexpr int M = _M * 8;
        
        for(int y = 0; y < N; y++) {
            for(int x = 0; x < M; x++) {
                float sum = 0;
                for(int k = 0; k < K; k++) {
//                    sum += i_ref[y*K + k] * i_ref[(N*K) + x*K + k];
                    sum += i_ref[y * K + k] * i_ref[(N * K) + x * K + k];
                }
                o_ref[y*M + x] = sum;
            }
        }
    }
};

template <typename T>
struct test_mma_AtB {
    template<int H, int W, int NW, typename K> using valid = std::bool_constant<NW == 1 && (2*W*H+W*K::value+H*K::value)<=64>; // this is warp-level
    static inline const std::string test_identifier = "reg_mma_AtB";
    static inline const std::string kernel_identifier = "reg_mma_AtB";
    using dtype = float;
    /*
     A: KxN
     B: KxM
     C: NxM
     */
    template<int _N, int _M, int NW, typename _K>
    static void host_func(const std::vector<float> &i_ref, std::vector<float> &o_ref) {
        constexpr int K = _K::value * 8;
        constexpr int N = _N * 8;
        constexpr int M = _M * 8;
        
        for(int y = 0; y < N; y++) {
            for(int x = 0; x < M; x++) {
                float sum = 0;
                for(int k = 0; k < K; k++) {
                    sum += i_ref[k * N + y] * i_ref[(N * K) + k * M + x];
                }
                o_ref[y*M + x] = sum;
            }
        }
    }
};

template <typename T>
struct test_mma_AtBt {
    template<int H, int W, int NW, typename K> using valid = std::bool_constant<NW == 1 && (2*W*H+W*K::value+H*K::value)<=64>; // this is warp-level
    static inline const std::string test_identifier = "reg_mma_AtBt";
    static inline const std::string kernel_identifier = "reg_mma_AtBt";
    using dtype = float;
    /*
     A: KxN
     B: MxK
     C: NxM
     */
    template<int _N, int _M, int NW, typename _K>
    static void host_func(const std::vector<float> &i_ref, std::vector<float> &o_ref) {
        constexpr int K = _K::value * 8;
        constexpr int N = _N * 8;
        constexpr int M = _M * 8;
        
        for(int y = 0; y < N; y++) {
            for(int x = 0; x < M; x++) {
                float sum = 0;
                for(int k = 0; k < K; k++) {
                    sum += i_ref[k * N + y] * i_ref[(N * K) + x * K + k];
                }
                o_ref[y*M + x] = sum;
            }
        }
    }
};



template<typename test, int MAX_H=8, int MAX_W=8, int NUM_WORKERS=1, typename... args> using mma_sweep_size = loop_h<mma_wrapper_2d, test, MAX_H, MAX_W, NUM_WORKERS, MAX_H, args...>;
template<typename test, int MAX_H=8, int MAX_W=8, typename... args> using mma_sweep_size_warp = mma_sweep_size<test, MAX_H, MAX_W, 1, args...>;

void warp::reg::tile::mma::tests(MTL::Device* device, MTL::CommandQueue* command_queue, test_data &results) {
    std::cout << "\n ----- Starting ops/warp/register/tile/mma tests! -----\n" << std::endl;
    constexpr int SIZE = INTENSITY_1 ? 2  :
                         INTENSITY_2 ? 4  :
                         INTENSITY_3 ? 8  :
                         INTENSITY_4 ? 16 : -1;

    mma_sweep_size_warp<test_mma_AB<float>, SIZE, SIZE, std::integral_constant<int, 1>>::run(device, command_queue, results);
    mma_sweep_size_warp<test_mma_AB<float>, SIZE, SIZE, std::integral_constant<int, 2>>::run(device, command_queue, results);
    mma_sweep_size_warp<test_mma_AB<float>, SIZE, SIZE, std::integral_constant<int, 3>>::run(device, command_queue, results);
    mma_sweep_size_warp<test_mma_AB<float>, SIZE, SIZE, std::integral_constant<int, 4>>::run(device, command_queue, results);

    mma_sweep_size_warp<test_mma_ABt<float>, SIZE, SIZE, std::integral_constant<int, 1>>::run(device, command_queue, results);
    mma_sweep_size_warp<test_mma_ABt<float>, SIZE, SIZE, std::integral_constant<int, 2>>::run(device, command_queue, results);
    mma_sweep_size_warp<test_mma_ABt<float>, SIZE, SIZE, std::integral_constant<int, 3>>::run(device, command_queue, results);
    mma_sweep_size_warp<test_mma_ABt<float>, SIZE, SIZE, std::integral_constant<int, 4>>::run(device, command_queue, results);

    mma_sweep_size_warp<test_mma_AtB<float>, SIZE, SIZE, std::integral_constant<int, 1>>::run(device, command_queue, results);
    mma_sweep_size_warp<test_mma_AtB<float>, SIZE, SIZE, std::integral_constant<int, 2>>::run(device, command_queue, results);
    mma_sweep_size_warp<test_mma_AtB<float>, SIZE, SIZE, std::integral_constant<int, 3>>::run(device, command_queue, results);
    mma_sweep_size_warp<test_mma_AtB<float>, SIZE, SIZE, std::integral_constant<int, 4>>::run(device, command_queue, results);
    
    mma_sweep_size_warp<test_mma_AtBt<float>, SIZE, SIZE, std::integral_constant<int, 1>>::run(device, command_queue, results);
    mma_sweep_size_warp<test_mma_AtBt<float>, SIZE, SIZE, std::integral_constant<int, 2>>::run(device, command_queue, results);
    mma_sweep_size_warp<test_mma_AtBt<float>, SIZE, SIZE, std::integral_constant<int, 3>>::run(device, command_queue, results);
    mma_sweep_size_warp<test_mma_AtBt<float>, SIZE, SIZE, std::integral_constant<int, 4>>::run(device, command_queue, results);
}

////
//int main() {
////    should_write_outputs = 1;
//    NS::Error** error;
//    MTL::CopyAllDevices();
//    MTL::Device* device = MTL::CreateSystemDefaultDevice();
//    MTL::CommandQueue* command_queue = device->newCommandQueue();
//    test_data results;
//    warp::reg::tile::mma::tests(device, command_queue, results);
//}


#endif


