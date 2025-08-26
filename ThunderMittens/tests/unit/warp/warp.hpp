#include "testing_flags.hpp"

#ifdef TEST_WARP

#include "testing_commons.hpp"

#include "memory/memory.hpp"
#include "register/register.hpp"
#include "shared/shared.hpp"

namespace warp {

void tests(MTL::Device* device, MTL::CommandQueue* command_queue, test_data &results);


}

#endif
