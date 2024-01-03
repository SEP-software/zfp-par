find_package(PythonInterp REQUIRED)
find_package(PythonLibs REQUIRED)
set(python_install_lib_dir "lib/python${PYTHON_VERSION_MAJOR}.${PYTHON_VERSION_MINOR}/site-packages/")

install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/_zfp_par.py ${CMAKE_CURRENT_SOURCE_DIR}/__init__.py DESTINATION ${python_install_lib_dir})
