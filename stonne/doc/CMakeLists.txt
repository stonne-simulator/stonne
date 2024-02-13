find_package(Doxygen)
if (NOT Doxygen_FOUND)
  message(WARNING "Doxygen not found, not building documentation")
  return()
endif ()

set(DOXYGEN_EXTRACT_ALL YES)
set(DOXYGEN_BUILTIN_STL_SUPPORT YES)
set(DOXYGEN_JAVADOC_AUTO_BRIEF YES)
set(DOXYGEN_DISABLE_INDEX YES)
set(DOXYGEN_GENERATE_TREEVIEW YES)
set(DOXYGEN_FULL_SIDEBAR NO)

doxygen_add_docs(doc
    stonne/
    WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}/include"
)

message(STATUS "Doxygen build target added")

