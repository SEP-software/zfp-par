cmake_minimum_required(VERSION 3.9)
project(zfpy)

install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/_zfp_par.py ${CMAKE_CURRENT_SOURCE_DIR}/__init__.py DESTINATION "zfpy" COMPONENT zfpy)
