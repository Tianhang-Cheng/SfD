cmake_minimum_required(VERSION 2.8)
# project(CeresTest)

set(ceres_DIR ~/Software/LocalInstall/usr/local/lib/cmake/Ceres/)
find_package(ceres REQUIRED)
include_directories(${EIGEN_INCLUDE_DIR})

# Find python and Boost - both are required dependencies
find_package(PythonLibs 3 REQUIRED)
find_package(Boost COMPONENTS python3 REQUIRED)

if(APPLE)
    set(CMAKE_SHARED_LIBRARY_SUFFIX ".so")
endif(APPLE)

# Without this, any build libraries automatically have names "lib{x}.so"
# set(CMAKE_SHARED_MODULE_PREFIX "")

# add_executable(
#   hello_world
#   hello_world.cc
# )
# target_link_libraries(
#   hello_world
#   ceres
# )
add_library(pyceres_ba SHARED bundle_adjustment.cc)
target_link_libraries(pyceres_ba ${Boost_LIBRARIES} ${PYTHON_LIBRARIES} ceres)