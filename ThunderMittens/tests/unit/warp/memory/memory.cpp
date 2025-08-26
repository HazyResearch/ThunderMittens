#include "memory.hpp"

#ifdef TEST_WARP_MEMORY

void warp::memory::tests(MTL::Device* device, MTL::CommandQueue* command_queue, test_data &results) {
    std::cout << "\n -------------------- Starting ops/warp/memory tests! --------------------\n" << std::endl;
#ifdef TEST_WARP_MEMORY_TILE
    warp::memory::tile::tests(device, command_queue, results);
#endif
#ifdef TEST_WARP_MEMORY_VEC
    warp::memory::vec::tests(device, command_queue, results);
#endif
}

#endif
