#===================== FINDING LIBRARIES ======================#
find_package(Catch2 CONFIG REQUIRED)

#======================== ENABLE TESTS ========================#
include(CTest)
include(Catch)
enable_testing()

#=========================== TESTS ============================#
set(TESTUNITS
    Conv
    DenseGEMM
    SparseDense
    InnerProductGEMM
    OuterProductGEMM
    GustavsonGEMM
)

foreach (testunit ${TESTUNITS})
  add_executable(${testunit} sim/${testunit}.cpp)
  target_link_libraries(${testunit} PRIVATE stonne Catch2::Catch2 Catch2::Catch2WithMain)
  catch_discover_tests(${testunit}
      TEST_SPEC [test][small]
      TEST_SUFFIX .test.small
  )
  catch_discover_tests(${testunit}
      TEST_SPEC [test][large]
      TEST_SUFFIX .test.large
  )
  catch_discover_tests(${testunit}
      TEST_SPEC [benchmark][small]
      TEST_SUFFIX .benchmark.small
      EXTRA_ARGS --benchmark-samples 10
  )
  catch_discover_tests(${testunit}
      TEST_SPEC [benchmark][large]
      TEST_SUFFIX .benchmark.large
      EXTRA_ARGS --benchmark-samples 10
  )
endforeach ()