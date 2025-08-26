#include "testing_flags.hpp"
#ifdef TEST_GROUP_MEMORY_VEC_SHARED_TO_REGISTER
#include "testing_commons.hpp"
#include "testing_utils.hpp"




namespace group {
namespace memory {
namespace vec {
namespace shared_to_register {

void tests(MTL::Device* device, MTL::CommandQueue* command_queue, test_data &results);

}
}
}
}

#endif
