#include "testing_flags.hpp"

#ifdef TEST_GROUP_SHARED_TILE

#include "testing_commons.hpp"

#include "conversions.hpp"
#include "maps.hpp"
#include "reductions.hpp"

namespace group {
namespace shared {
namespace tile {

void tests(MTL::Device* device, MTL::CommandQueue* command_queue, test_data &results);

}
}
}

#endif
