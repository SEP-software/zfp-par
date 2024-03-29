if (CMAKE_VERSION VERSION_GREATER_EQUAL 3.27.0)
    cmake_policy(SET CMP0148 OLD)
endif ()

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_LIST_DIR}/scikit-build-cmake)
include(UseCython)
include(FindPythonExtensions)
include(FindNumPy)

find_package(PythonInterp REQUIRED)
find_package(PythonLibs REQUIRED)
find_package(PythonExtensions REQUIRED)
find_package(Cython 0.28 REQUIRED) # >= v0.28 required for const memoryview support
find_package(NumPy REQUIRED)

include_directories(${ZFP_SOURCE_DIR}/include)
include_directories(${NumPy_INCLUDE_DIR})
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

add_cython_target(zfpy_c zfpy_c.pyx C PY3)
add_library(zfpy_c MODULE ${zfpy_c})
target_link_libraries(zfpy_c zfp)
python_extension_module(zfpy_c)

# Build to the current binary dir to avoid conflicts with other libraries named zfp
set(PYLIB_BUILD_DIR "${CMAKE_BINARY_DIR}/bin" CACHE PATH "Directory where zfp python library will be built")
set_target_properties(zfpy_c PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${PYLIB_BUILD_DIR})

# Install to the typical python module directory
set(python_install_lib_dir "lib/python${PYTHON_VERSION_MAJOR}.${PYTHON_VERSION_MINOR}/site-packages/")

if(BUILD_PIP)
    message("IN THE REDIRECTION")
    set(python_install_lib_dir "zfpy")
else(BUILD_PIP)
    message("NO REDIRECTION???")
endif()
install(TARGETS zfpy_c LIBRARY DESTINATION ${python_install_lib_dir})
