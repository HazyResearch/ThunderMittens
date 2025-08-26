#include "testing_flags.hpp"

#ifdef TEST_GROUP_SHARED

#include "testing_commons.hpp"

#include "tile/tile.hpp"
#include "vec/vec.hpp"

namespace group {
namespace shared {

void tests(MTL::Device* device, MTL::CommandQueue* command_queue, test_data &results);


}
}

#endif
