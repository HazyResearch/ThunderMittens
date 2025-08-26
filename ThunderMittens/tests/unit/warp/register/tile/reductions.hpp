#include "testing_flags.hpp"

#ifdef TEST_WARP_REGISTER_TILE_REDUCTIONS

#include "testing_commons.hpp"

namespace warp {
namespace reg {
namespace tile {
namespace reductions {

void tests(MTL::Device* device, MTL::CommandQueue* command_queue, test_data &results);

}
}
}
}

#endif



