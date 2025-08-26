// global_to_register.hpp
#include "testing_flags.hpp"
#ifdef TEST_WARP_MEMORY_TILE_SHARED_TO_REGISTER
#include "MetalSingle.hpp"


#include "testing_commons.hpp"
#include "testing_utils.hpp"





namespace warp {
namespace memory {
namespace tile {
namespace shared_to_register {

void tests(MTL::Device* device, MTL::CommandQueue* command_queue, test_data &results);

}
}
}
}

#endif
