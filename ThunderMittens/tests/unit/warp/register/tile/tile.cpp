#include "tile.hpp"

#ifdef TEST_WARP_REGISTER_TILE

void warp::reg::tile::tests(MTL::Device* device, MTL::CommandQueue* command_queue, test_data &results) {
    std::cout << "\n --------------- Starting ops/warp/register/tile tests! ---------------\n" << std::endl;
#ifdef TEST_WARP_REGISTER_TILE_CONVERSIONS
    warp::reg::tile::conversions::tests(device, command_queue, results);
#endif
#ifdef TEST_WARP_REGISTER_TILE_MAPS
    warp::reg::tile::maps::tests(device, command_queue, results);
#endif
#ifdef TEST_WARP_REGISTER_TILE_MMA
    warp::reg::tile::mma::tests(device, command_queue, results);
#endif
#ifdef TEST_WARP_REGISTER_TILE_REDUCTIONS
    warp::reg::tile::reductions::tests(device, command_queue, results);
#endif

}

#endif
