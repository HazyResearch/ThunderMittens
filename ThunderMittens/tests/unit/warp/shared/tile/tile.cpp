#include "tile.hpp"

#ifdef TEST_WARP_SHARED_TILE

void warp::shared::tile::tests(MTL::Device* device, MTL::CommandQueue* command_queue, test_data &results) {
    std::cout << "\n --------------- Starting ops/warp/shared/tile tests! ---------------\n" << std::endl;
#ifdef TEST_WARP_SHARED_TILE_CONVERSIONS
    warp::shared::tile::conversions::tests(device, command_queue, results);
#endif
#ifdef TEST_WARP_SHARED_TILE_MAPS
    warp::shared::tile::maps::tests(device, command_queue, results);
#endif
#ifdef TEST_WARP_SHARED_TILE_REDUCTIONS
    warp::shared::tile::reductions::tests(device, command_queue, results);
#endif

}

#endif
