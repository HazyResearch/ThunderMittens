#include "testing_flags.hpp"

#ifdef TEST_GROUP_MEMORY_TILE

#include "testing_commons.hpp"

#include "global_to_register.hpp"
#include "global_to_shared.hpp"
#include "shared_to_register.hpp"

namespace group {
namespace memory {
namespace tile {

void tests(MTL::Device* device, MTL::CommandQueue* command_queue, test_data &results);

}
}
}

#endif


