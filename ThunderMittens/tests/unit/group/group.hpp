#include "testing_flags.hpp"

#ifdef TEST_GROUP

#include "testing_commons.hpp"

#include "memory/memory.hpp"
#include "shared/shared.hpp"

namespace group {

void tests(MTL::Device* device, MTL::CommandQueue* command_queue, test_data &results);


}

#endif
