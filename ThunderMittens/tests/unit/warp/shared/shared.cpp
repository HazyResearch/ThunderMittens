#include "shared.hpp"

#ifdef TEST_WARP_SHARED

void warp::shared::tests(MTL::Device* device, MTL::CommandQueue* command_queue, test_data &results) {
    std::cout << "\n --------------- Starting ops/warp/shared tests! ---------------\n" << std::endl;
#ifdef TEST_WARP_SHARED_TILE
    warp::shared::tile::tests(device, command_queue, results);
#endif
#ifdef TEST_WARP_SHARED_VEC
    warp::shared::vec::tests(device, command_queue, results);
#endif

}

#endif
