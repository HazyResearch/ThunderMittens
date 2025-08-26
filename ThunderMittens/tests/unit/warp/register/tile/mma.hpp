#include "testing_flags.hpp"

#ifdef TEST_WARP_REGISTER_TILE_MMA

#include "testing_commons.hpp"

namespace warp {
namespace reg {
namespace tile {
namespace mma {

void tests(MTL::Device* device, MTL::CommandQueue* command_queue, test_data &results);

}
}
}
}

#endif


