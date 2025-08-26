#include "tile.hpp"

#ifdef TEST_WARP_MEMORY_TILE

void warp::memory::tile::tests(MTL::Device* device, MTL::CommandQueue* command_queue, test_data &results) {
    std::cout << "\n --------------- Starting ops/warp/memory/tile tests! ---------------\n" << std::endl;
#ifdef TEST_WARP_MEMORY_TILE_GLOBAL_TO_REGISTER
    warp::memory::tile::global_to_register::tests(device, command_queue, results);
#endif
#ifdef TEST_WARP_MEMORY_TILE_GLOBAL_TO_SHARED
    warp::memory::tile::global_to_shared::tests(device, command_queue, results);
#endif
#ifdef TEST_WARP_MEMORY_TILE_SHARED_TO_REGISTER
    warp::memory::tile::shared_to_register::tests(device, command_queue, results);
#endif

}

#endif

