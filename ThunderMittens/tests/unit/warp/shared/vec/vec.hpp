#include "testing_flags.hpp"

#ifdef TEST_WARP_SHARED_VEC

#include "testing_commons.hpp"

#include "conversions.hpp"
#include "maps.hpp"
#include "reductions.hpp"

namespace warp {
namespace shared {
namespace vec {

void tests(MTL::Device* device, MTL::CommandQueue* command_queue, test_data &results);

}
}
}

#endif
