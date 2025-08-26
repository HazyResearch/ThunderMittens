#include "testing_flags.hpp"

#ifdef TEST_WARP_REGISTER_VEC

#include "testing_commons.hpp"

#include "conversions.hpp"
#include "maps.hpp"
#include "reductions.hpp"

namespace warp {
namespace reg {
namespace vec {

void tests(MTL::Device* device, MTL::CommandQueue* command_queue, test_data &results);

}
}
}

#endif
