cmake_minimum_required(VERSION 3.20 FATAL_ERROR)
cmake_policy(VERSION 3.20)
if(POLICY CMP0144)
  cmake_policy(SET CMP0144 NEW)
endif()

project(CPP2PY VERSION 2.0.0 LANGUAGES CXX)

#--------------------------------------------------------
#   Basic setup
#--------------------------------------------------------
get_directory_property(IS_SUBPROJECT PARENT_DIRECTORY)
include(CheckIncludeFile)
include(CheckIncludeFileCXX)

# default to Release build type
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Type of build" FORCE)
endif()
message(STATUS "-------- BUILD-TYPE: ${CMAKE_BUILD_TYPE} -------------")

# build static library
set(BUILD_SHARED_LIBS OFF)

# check CMAKE_INSTALL_PREFIX : must be provided and absolute
if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT OR (NOT IS_ABSOLUTE ${CMAKE_INSTALL_PREFIX}))
  message(FATAL_ERROR "CMAKE_INSTALL_PREFIX must be specified and must be an absolute path.\n There is no default.\n Current value : ${CMAKE_INSTALL_PREFIX}\n.  e.g. ... -DCMAKE_INSTALL_PREFIX=$HOME/triqs_install")
endif()
message(STATUS "Installation directory will be ${CMAKE_INSTALL_PREFIX}")
set(CPP2PY_BINARY_DIR ${PROJECT_BINARY_DIR} CACHE STRING "Binary directory of the Cpp2Py Project")

# add ./cmake to the path for module
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

# log info
message( STATUS "-------- cpp2py version and git hash detection -------------")

find_package(Git)

# get the Git Hash
execute_process(COMMAND ${GIT_EXECUTABLE} rev-parse HEAD
  WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
  OUTPUT_VARIABLE   CPP2PY_GIT_HASH
  OUTPUT_STRIP_TRAILING_WHITESPACE
)

message(STATUS "Cpp2py version : ${CPP2PY_VERSION}")
message(STATUS "Git hash : ${CPP2PY_GIT_HASH}")

#--------------------------------------------------------
#   Libclang
#--------------------------------------------------------
message(STATUS "-------- LibClang detection -------------")

find_package(LibClang)

message(STATUS "LibClang location: ${LIBCLANG_LOCATION}")
message(STATUS "LibClang additional flags: ${LIBCLANG_CXX_FLAGS}")

# allow user to modify
set(LIBCLANG_CXX_FLAGS "${LIBCLANG_CXX_FLAGS}" CACHE STRING "Additional flags to be passed to libclang when parsing with clang")
set(LIBCLANG_LOCATION "${LIBCLANG_LOCATION}" CACHE STRING "Location of the libclang library")

#--------------------------------------------------------
#   Python and Numpy
#--------------------------------------------------------
message(STATUS "-------- Python detection -------------")

find_package(Python REQUIRED COMPONENTS Interpreter Development NumPy)

# check required python packages
function(exec_python_command command_str output_var_name)
  execute_process(COMMAND ${Python_EXECUTABLE} -c ${command_str}
    OUTPUT_VARIABLE res RESULT_VARIABLE returncode OUTPUT_STRIP_TRAILING_WHITESPACE)
  if(NOT returncode EQUAL 0)
    message(FATAL_ERROR "The command : ${command_str} \n did not run properly in the Python interpreter. Check your python installation.")
  endif()
  set(${output_var_name} ${res} PARENT_SCOPE)
endfunction()

exec_python_command("import mako.template" nulle)
exec_python_command("import numpy" nulle)
exec_python_command("import scipy" nulle)
if(Build_Documentation)
  exec_python_command("import clang.cindex" nulle)
endif()

# set variables used in the current and in dependent projects
set(CPP2PY_PYTHON_EXECUTABLE ${Python_EXECUTABLE} CACHE FILEPATH "Python executable")
if(Python_SOABI)
  set(CPP2PY_PYTHON_MODULE_EXT .${Python_SOABI}.so CACHE STRING "Python module extension")
else()
  set(CPP2PY_PYTHON_MODULE_EXT .so CACHE STRING "Python module extension")
endif()
if(BUILD_DEBIAN_PACKAGE)
  set(CPP2PY_PYTHON_LIB_DEST_ROOT lib/python${Python_VERSION_MAJOR}/dist-packages CACHE PATH "Python library destination")
else()
  set(CPP2PY_PYTHON_LIB_DEST_ROOT lib/python${Python_VERSION_MAJOR}.${Python_VERSION_MINOR}/site-packages CACHE PATH "Python library destination")
endif()

# output the python variables
message(STATUS "Python_INCLUDE_DIRS: " ${Python_INCLUDE_DIRS})
message(STATUS "Python_NumPy_INCLUDE_DIRS: " ${Python_NumPy_INCLUDE_DIRS})
message(STATUS "Python_NumPy_VERSION: " ${Python_NumPy_VERSION})
message(STATUS "CPP2PY_PYTHON_MODULE_EXT: " ${CPP2PY_PYTHON_MODULE_EXT})
message(STATUS "Python modules install path: ${CMAKE_INSTALL_PREFIX}/${CPP2PY_PYTHON_LIB_DEST_ROOT}")

# define the python_and_numpy interface target
add_library(python_and_numpy INTERFACE)
add_library(cpp2py::python_and_numpy ALIAS python_and_numpy)
target_link_libraries(python_and_numpy INTERFACE Python::Python Python::NumPy)
install(TARGETS python_and_numpy EXPORT Cpp2PyTargets)

# set numpy related macros (C API changed with v1.7.0)
if(Python_NumPy_VERSION VERSION_LESS "1.7.0")
  set_property(TARGET python_and_numpy PROPERTY INTERFACE_COMPILE_DEFINITIONS PYTHON_NUMPY_VERSION_LT_17)
endif()

if(Python_NumPy_VERSION VERSION_GREATER "1.8.0")
  set_property(TARGET python_and_numpy PROPERTY INTERFACE_COMPILE_DEFINITIONS NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION)
endif()

# Provide GNU Installation directories
include(GNUInstallDirs)

#--------------------------------------------------------
#   Build cpp2py
#--------------------------------------------------------

add_subdirectory(bin)
add_subdirectory(cmake)
add_subdirectory(c++/cpp2py)

# python modules
add_subdirectory(cpp2py)
add_subdirectory(cpp2rst)
add_subdirectory(cpp2cxx)

#--------------------------------------------------------
#   Libclang Warning
#--------------------------------------------------------
if (NOT LIBCLANG_LOCATION)
  message(STATUS "**************************************************************************************")
  message(STATUS "**************************          WARNING           ********************************")
  message(STATUS "   ")
  message(STATUS "    Can not find libclang                                                           ")
  message(STATUS "    You can use cpp2py to compile a code, but c++2py, c++2rst, c++2cxx will not work")
  message(STATUS "   ")
  message(STATUS "**************************          WARNING           ********************************")
  message(STATUS "**************************************************************************************")
endif()

#--------------------------------------------------------
#   Remind the user how to set up his/her variables
#--------------------------------------------------------
if(NOT IS_SUBPROJECT
  AND NOT CMAKE_INSTALL_PREFIX STREQUAL "/usr"
  AND NOT CMAKE_INSTALL_PREFIX STREQUAL "/usr/local"
)
  # Configure and install the file to source to setup the environment variables
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/cpp2pyvars.sh.in ${CMAKE_CURRENT_BINARY_DIR}/cpp2pyvars.sh @ONLY)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/cpp2py.modulefile.in ${CMAKE_CURRENT_BINARY_DIR}/cpp2py.modulefile @ONLY)

  install(FILES ${CMAKE_CURRENT_BINARY_DIR}/cpp2pyvars.sh DESTINATION share/Cpp2Py)
  message(STATUS "***************************************************************")
  message(STATUS "* Use :                                                        ")
  message(STATUS "*   source ${CMAKE_INSTALL_PREFIX}/share/Cpp2Py/cpp2pyvars.sh  ")
  message(STATUS "*   to set up the environment variables                        ")
  if(DEFINED ENV{MODULEPATH})
    message(STATUS "*                                                                ")
    message(STATUS "* Consider copying ${CMAKE_CURRENT_BINARY_DIR}/cpp2py.modulefile ")
    message(STATUS "*   into your environment module directories                     ")
  endif()
  message(STATUS "***************************************************************")
endif()

#--------------------------------------------------------
#   Uninstall target
#--------------------------------------------------------
if(NOT TARGET uninstall)
  configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake_uninstall.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake"
    IMMEDIATE @ONLY)

  add_custom_target(uninstall COMMAND ${CMAKE_COMMAND} -P ${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake)
endif()

#--------------------------------------------------------
#   Packaging
#--------------------------------------------------------
option(BUILD_DEBIAN_PACKAGE "Build a deb package" OFF)
if(BUILD_DEBIAN_PACKAGE)
  if(NOT CMAKE_INSTALL_PREFIX STREQUAL "/usr")
    message(FATAL_ERROR "CMAKE_INSTALL_PREFIX must be /usr for packaging")
  endif()
  set(CPACK_PACKAGE_NAME cpp2py)
  set(CPACK_GENERATOR "DEB")
  set(CPACK_PACKAGE_VERSION ${CPP2PY_VERSION})
  set(CPACK_PACKAGE_CONTACT "https://github.com/TRIQS/cpp2py")
  execute_process(COMMAND dpkg --print-architecture OUTPUT_VARIABLE CPACK_DEBIAN_PACKAGE_ARCHITECTURE OUTPUT_STRIP_TRAILING_WHITESPACE)
  set(CPACK_DEBIAN_PACKAGE_DEPENDS "python, python-mako, python-numpy, python-h5py, python-scipy")
  set(CPACK_DEBIAN_PACKAGE_CONFLICTS "triqs (>= 2.1)")
  SET(CPACK_DEBIAN_PACKAGE_SHLIBDEPS ON)
  SET(CPACK_DEBIAN_PACKAGE_GENERATE_SHLIBS ON)
  include(CPack)
endif()
