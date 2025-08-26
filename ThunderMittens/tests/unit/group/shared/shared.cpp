#include "shared.hpp"

#ifdef TEST_GROUP_SHARED

void group::shared::tests(MTL::Device* device, MTL::CommandQueue* command_queue, test_data &results) {
    std::cout << "\n --------------- Starting ops/group/shared tests! ---------------\n" << std::endl;
#ifdef TEST_GROUP_SHARED_TILE
    group::shared::tile::tests(device, command_queue, results);
#endif
#ifdef TEST_GROUP_SHARED_VEC
    group::shared::vec::tests(device, command_queue, results);
#endif

}

#endif
