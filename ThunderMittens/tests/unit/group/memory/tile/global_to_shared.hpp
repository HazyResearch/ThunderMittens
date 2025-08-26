#include "testing_flags.hpp"
#ifdef TEST_GROUP_MEMORY_TILE_GLOBAL_TO_SHARED
#include "testing_commons.hpp"
#include "testing_utils.hpp"




namespace group {
namespace memory {
namespace tile {
namespace global_to_shared {

void tests(MTL::Device* device, MTL::CommandQueue* command_queue, test_data &results);

}
}
}
}

#endif
