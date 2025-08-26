#include "warp.hpp"

#ifdef TEST_WARP

void warp::tests(MTL::Device* device, MTL::CommandQueue* command_queue, test_data &results) {
    std::cout << "\n --------------- Starting ops/warp tests! ---------------\n" << std::endl;
#ifdef TEST_WARP_MEMORY
    warp::memory::tests(device, command_queue, results);
#endif
#ifdef TEST_WARP_REGISTER
    warp::reg::tests(device, command_queue, results);
#endif
#ifdef TEST_WARP_SHARED
    warp::shared::tests(device, command_queue, results);
#endif

}

#endif
