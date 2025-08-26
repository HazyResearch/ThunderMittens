#include "tile.hpp"

#ifdef TEST_GROUP_SHARED_TILE

void group::shared::tile::tests(MTL::Device* device, MTL::CommandQueue* command_queue, test_data &results) {
    std::cout << "\n --------------- Starting ops/groups/shared/tile tests! ---------------\n" << std::endl;
#ifdef TEST_GROUP_SHARED_TILE_CONVERSIONS
    group::shared::tile::conversions::tests(device, command_queue, results);
#endif
#ifdef TEST_GROUP_SHARED_TILE_MAPS
    group::shared::tile::maps::tests(device, command_queue, results);
#endif
#ifdef TEST_GROUP_SHARED_TILE_REDUCTIONS
    group::shared::tile::reductions::tests(device, command_queue, results);
#endif

}

#endif
