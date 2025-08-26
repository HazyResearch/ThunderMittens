#include "tile.hpp"

#ifdef TEST_GROUP_MEMORY_TILE

void group::memory::tile::tests(MTL::Device* device, MTL::CommandQueue* command_queue, test_data &results) {
    std::cout << "\n --------------- Starting ops/group/memory/tile tests! ---------------\n" << std::endl;
#ifdef TEST_GROUP_MEMORY_TILE_GLOBAL_TO_REGISTER
    group::memory::tile::global_to_register::tests(device, command_queue, results);
#endif
#ifdef TEST_GROUP_MEMORY_TILE_GLOBAL_TO_SHARED
    group::memory::tile::global_to_shared::tests(device, command_queue, results);
#endif
#ifdef TEST_GROUP_MEMORY_TILE_SHARED_TO_REGISTER
    group::memory::tile::shared_to_register::tests(device, command_queue, results);
#endif

}

#endif

