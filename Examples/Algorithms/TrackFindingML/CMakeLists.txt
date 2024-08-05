set(SOURCES 
  src/AmbiguityResolutionML.cpp
  src/AmbiguityResolutionMLAlgorithm.cpp
  src/AmbiguityResolutionMLDBScanAlgorithm.cpp   
  src/SeedFilterMLAlgorithm.cpp
)

<<<<<<< HEAD
if(ACTS_BUILD_PLUGIN_MLPACK)
  list(APPEND SOURCES 
    src/AmbiguityResolutionMLDBScanAlgorithm.cpp   
    src/SeedFilterMLAlgorithm.cpp
  )
endif()

=======
>>>>>>> upstream/main
add_library(
  ActsExamplesTrackFindingML SHARED
  ${SOURCES}
)

target_include_directories(
  ActsExamplesTrackFindingML
  PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)

target_link_libraries(
  ActsExamplesTrackFindingML
  PUBLIC
    ActsCore
    ActsPluginOnnx
    ActsExamplesFramework
)

install(
  TARGETS ActsExamplesTrackFindingML
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
)
