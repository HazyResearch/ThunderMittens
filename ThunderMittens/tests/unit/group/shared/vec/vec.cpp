#include "vec.hpp"

#ifdef TEST_GROUP_SHARED_VEC

void group::shared::vec::tests(MTL::Device* device, MTL::CommandQueue* command_queue, test_data &results) {
    std::cout << "\n --------------- Starting ops/group/shared/vec tests! ---------------\n" << std::endl;
#ifdef TEST_GROUP_SHARED_VEC_CONVERSIONS
    group::shared::vec::conversions::tests(device, command_queue, results);
#endif
#ifdef TEST_GROUP_SHARED_VEC_MAPS
    group::shared::vec::maps::tests(device, command_queue, results);
#endif

}

#endif
