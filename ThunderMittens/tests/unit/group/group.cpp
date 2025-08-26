#include "group.hpp"

#ifdef TEST_GROUP

void group::tests(MTL::Device* device, MTL::CommandQueue* command_queue, test_data &results) {
    std::cout << "\n --------------- Starting ops/group tests! ---------------\n" << std::endl;
#ifdef TEST_GROUP_MEMORY
    group::memory::tests(device, command_queue, results);
#endif
#ifdef TEST_GROUP_SHARED
    group::shared::tests(device, command_queue, results);
#endif

}

#endif
