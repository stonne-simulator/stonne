#===================== FINDING LIBRARIES ======================#
find_package(Catch2 CONFIG REQUIRED)

#======================== ENABLE TESTS ========================#
include(CTest)
include(Catch)
enable_testing()

#=========================== TESTS ============================#
set(SMALL_TESTUNITS
    small_conv
    small_densegemm
    small_sparsedense
    small_innerproductgemm
    small_outerproductgemm
    small_gustavsongemm
)

set(LARGE_TESTUNITS
    large_conv
    large_densegemm
    large_innerproductgemm
    large_outerproductgemm
    large_gustavsongemm
)

foreach (testunit ${SMALL_TESTUNITS})
  add_executable(${testunit} sim/${testunit}.cpp)
  target_link_libraries(${testunit} PRIVATE stonne Catch2::Catch2 Catch2::Catch2WithMain)
  catch_discover_tests(${testunit}
      TEST_SPEC [test]
      TEST_SUFFIX .test.small
  )
  catch_discover_tests(${testunit}
      TEST_SPEC [benchmark]
      TEST_SUFFIX .benchmark.small
      EXTRA_ARGS --benchmark-samples 10
  )
endforeach ()

foreach (testunit ${LARGE_TESTUNITS})
  add_executable(${testunit} sim/${testunit}.cpp)
  target_link_libraries(${testunit} PRIVATE stonne Catch2::Catch2 Catch2::Catch2WithMain)
  catch_discover_tests(${testunit}
      TEST_SPEC [test]
      TEST_SUFFIX .test.large
  )
  catch_discover_tests(${testunit}
      TEST_SPEC [benchmark]
      TEST_SUFFIX .benchmark.large
      EXTRA_ARGS --benchmark-samples 10
  )
endforeach ()