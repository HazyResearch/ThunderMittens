#include "testing_flags.hpp"
#include "testing_commons.hpp"

#ifdef ENABLE_TESTS

#ifdef TEST_WARP
#include "warp/warp.hpp"
#endif
#ifdef TEST_GROUP
#include "group/group.hpp"
#endif

int main(int argc, char **argv) {

    should_write_outputs = argc>1; // write outputs if user says so

    test_data data;
    NS::Error** error;
    MTL::CopyAllDevices();
    MTL::Device* device = MTL::CreateSystemDefaultDevice();
    MTL::CommandQueue* command_queue = device->newCommandQueue();
    test_data results;
    
#ifdef TEST_WARP
    warp::tests(device, command_queue, data);
#endif
#ifdef TEST_GROUP
    group::tests(device, command_queue, data);
#endif

    std::cout << "\n ------------------------------     Summary     ------------------------------\n"  << std::endl;

    std::cout << "Failed tests:\n";
    int passes = 0, fails = 0, invalids = 0;
    for(auto it = data.begin(); it != data.end(); it++) {
        if(it->result == test_result::PASSED)  passes++;
        if(it->result == test_result::INVALID) invalids++;
        if(it->result == test_result::FAILED) {
            fails++;
            std::cout << it->label << std::endl;
        }
    }
    if(fails == 0) std::cout << "ALL TESTS PASSED!\n";
    std::cout << std::endl;

    std::cout << invalids << " tests skipped (this is normal, and refers to tests that cannot be compiled due to invalid template parameters.)\n";
    std::cout << passes   << " tests passed\n";
    std::cout << fails    << " tests failed\n";

    return 0;
}

#endif
