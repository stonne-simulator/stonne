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

add_executable(small_densegemm sim/small_densegemm.cpp)
target_link_libraries(small_densegemm PRIVATE stonne Catch2::Catch2 Catch2::Catch2WithMain)
catch_discover_tests(small_densegemm
    TEST_SPEC [test]
    TEST_SUFFIX .test
)
catch_discover_tests(small_densegemm
    TEST_SPEC [benchmark]
    TEST_SUFFIX .benchmark.small
    EXTRA_ARGS --benchmark-samples 5
)

add_executable(large_densegemm sim/large_densegemm.cpp)
target_link_libraries(large_densegemm PRIVATE stonne Catch2::Catch2 Catch2::Catch2WithMain)
catch_discover_tests(large_densegemm
    TEST_SPEC [test]
    TEST_SUFFIX .test
)
catch_discover_tests(large_densegemm
    TEST_SPEC [benchmark]
    TEST_SUFFIX .benchmark.large
    EXTRA_ARGS --benchmark-samples 3
)

add_executable(small_sparsedense sim/small_sparsedense.cpp)
target_link_libraries(small_sparsedense PRIVATE stonne Catch2::Catch2 Catch2::Catch2WithMain)
catch_discover_tests(small_sparsedense
    TEST_SPEC [test]
    TEST_SUFFIX .test
)
catch_discover_tests(small_sparsedense
    TEST_SPEC [benchmark]
    TEST_SUFFIX .benchmark.small
    EXTRA_ARGS --benchmark-samples 5
)

add_executable(small_sparsegemm sim/small_sparsegemm.cpp)
target_link_libraries(small_sparsegemm PRIVATE stonne Catch2::Catch2 Catch2::Catch2WithMain)
catch_discover_tests(small_sparsegemm
    TEST_SPEC [test]
    TEST_SUFFIX .test
)
catch_discover_tests(small_sparsegemm
    TEST_SPEC [benchmark]
    TEST_SUFFIX .benchmark.small
    EXTRA_ARGS --benchmark-samples 5
)


