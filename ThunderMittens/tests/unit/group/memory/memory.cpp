#include "memory.hpp"

#ifdef TEST_GROUP_MEMORY

void group::memory::tests(MTL::Device* device, MTL::CommandQueue* command_queue, test_data &results) {
    std::cout << "\n -------------------- Starting ops/group/memory tests! --------------------\n" << std::endl;
#ifdef TEST_GROUP_MEMORY_TILE
    group::memory::tile::tests(device, command_queue, results);
#endif
#ifdef TEST_GROUP_MEMORY_VEC
    group::memory::vec::tests(device, command_queue, results);
#endif
}

#endif
