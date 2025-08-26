#include "testing_flags.hpp"

#ifdef TEST_GROUP_SHARED_VEC

#include "testing_commons.hpp"

#include "conversions.hpp"
#include "maps.hpp"

namespace group {
namespace shared {
namespace vec {

void tests(MTL::Device* device, MTL::CommandQueue* command_queue, test_data &results);

}
}
}

#endif
