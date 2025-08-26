#include "vec.hpp"

#ifdef TEST_WARP_REGISTER_VEC

void warp::reg::vec::tests(MTL::Device* device, MTL::CommandQueue* command_queue, test_data &results) {
    std::cout << "\n --------------- Starting ops/warp/register/vec tests! ---------------\n" << std::endl;
#ifdef TEST_WARP_REGISTER_VEC_CONVERSIONS
    warp::reg::vec::conversions::tests(device, command_queue, results);
#endif
#ifdef TEST_WARP_REGISTER_VEC_MAPS
    warp::reg::vec::maps::tests(device, command_queue, results);
#endif
#ifdef TEST_WARP_REGISTER_VEC_REDUCTIONS
    warp::reg::vec::reductions::tests(device, command_queue, results);
#endif

}

#endif
