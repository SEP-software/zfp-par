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

add_cython_target(_zfpy _zfpy.pyx C PY3)
add_library(_zfpy MODULE ${_zfpy})
target_link_libraries(_zfpy zfp)
python_extension_module(_zfpy)

# Build to the current binary dir to avoid conflicts with other libraries named zfp
set(PYLIB_BUILD_DIR "${CMAKE_BINARY_DIR}/bin" CACHE PATH "Directory where zfp python library will be built")
set_target_properties(_zfpy PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${PYLIB_BUILD_DIR})

# Install to the typical python module directory
set(python_install_lib_dir "lib/python${PYTHON_VERSION_MAJOR}.${PYTHON_VERSION_MINOR}/site-packages/")
install(TARGETS _zfpy  LIBRARY DESTINATION ${python_install_lib_dir})

# Additional line to install the zfp.py file
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/_zfp.py ${CMAKE_CURRENT_SOURCE_DIR}/__init__.py DESTINATION ${python_install_lib_dir})
