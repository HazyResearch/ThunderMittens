#include "register.hpp"

#ifdef TEST_WARP_REGISTER

void warp::reg::tests(MTL::Device* device, MTL::CommandQueue* command_queue, test_data &results) {
    std::cout << "\n --------------- Starting ops/warp/register tests! ---------------\n" << std::endl;
#ifdef TEST_WARP_REGISTER_TILE
    warp::reg::tile::tests(device, command_queue, results);
#endif
#ifdef TEST_WARP_REGISTER_VEC
    warp::reg::vec::tests(device, command_queue, results);
#endif

}

#endif
