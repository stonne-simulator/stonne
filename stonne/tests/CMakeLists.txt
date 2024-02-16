#===================== FINDING LIBRARIES ======================#
find_package(Catch2 CONFIG REQUIRED)

#======================== ENABLE TESTS ========================#
include(CTest)
include(Catch)
enable_testing()

#=========================== TESTS ============================#
add_executable(small_conv sim/small_conv.cpp)
target_link_libraries(small_conv PRIVATE stonne Catch2::Catch2 Catch2::Catch2WithMain)
catch_discover_tests(small_conv
    TEST_SPEC [test]
    TEST_SUFFIX .test
)
catch_discover_tests(small_conv
    TEST_SPEC [benchmark]
    TEST_SUFFIX .benchmark.small
    EXTRA_ARGS --benchmark-samples 5
)

add_executable(large_conv sim/large_conv.cpp)
target_link_libraries(large_conv PRIVATE stonne Catch2::Catch2 Catch2::Catch2WithMain)
catch_discover_tests(large_conv
    TEST_SPEC [test]
    TEST_SUFFIX .test
)
catch_discover_tests(large_conv
    TEST_SPEC [benchmark]
    TEST_SUFFIX .benchmark.large
    EXTRA_ARGS --benchmark-samples 3
)


