#include "testing_flags.hpp"

#ifdef TEST_WARP_MEMORY

#include "testing_commons.hpp"

#include "tile/tile.hpp"
#include "vec/vec.hpp"

namespace warp {
namespace memory {

void tests(MTL::Device* device, MTL::CommandQueue* command_queue, test_data &results);

}
}

#endif
