cmake_minimum_required(VERSION 2.8.12)
project(cpp2py C CXX)
include(CheckIncludeFile)
include(CheckIncludeFileCXX)

# Default to Release build type
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Type of build" FORCE)
endif()
message( STATUS "-------- BUILD-TYPE: ${CMAKE_BUILD_TYPE} -------------")

# shared library. no option, use shared.
set(BUILD_SHARED_LIBS ON)

# Check CMAKE_INSTALL_PREFIX : must be provided and absolute
if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT OR (NOT IS_ABSOLUTE ${CMAKE_INSTALL_PREFIX}))
 message(FATAL_ERROR "CMAKE_INSTALL_PREFIX must be specified and must be an absolute path.\n There is no default.\n Current value : ${CMAKE_INSTALL_PREFIX}\n.  e.g. ... -DCMAKE_INSTALL_PREFIX=$HOME/triqs_install")
endif()
message(STATUS "Installation directory will be ${CMAKE_INSTALL_PREFIX}")

# add ./cmake to the path for module
list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake) 

# log info
message( STATUS "-------- cpp2py version and git hash detection -------------")

find_package(Git) 

# Get the version number from the latest annotated git tag
execute_process(COMMAND bash "-c" "${GIT_EXECUTABLE} describe | sed \"s/-\\([0-9]*\\)-.*/.\\1/g\""
 WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
 OUTPUT_VARIABLE   CPP2PY_VERSION
 OUTPUT_STRIP_TRAILING_WHITESPACE
)
if(NOT CPP2PY_VERSION)
  set(CPP2PY_VERSION 1.5)
endif()

# Get the Git Hash
execute_process(COMMAND ${GIT_EXECUTABLE} rev-parse HEAD
 WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
 OUTPUT_VARIABLE   CPP2PY_GIT_HASH
 OUTPUT_STRIP_TRAILING_WHITESPACE
)

message(STATUS "Cpp2py version : ${CPP2PY_VERSION}")
message(STATUS "Git hash : ${CPP2PY_GIT_HASH}")

# LIBCLANG : try to set LIBCLANG_LOCATION and LIBCLANG_CXX_FLAGS automatically
MESSAGE(STATUS "-------- LibClang detection -------------")

find_package(LibClang)

MESSAGE(STATUS "LibClang location: ${LIBCLANG_LOCATION}")
MESSAGE(STATUS "LibClang additional flags: ${LIBCLANG_CXX_FLAGS}")

# allow user to modify
set(LIBCLANG_CXX_FLAGS "${LIBCLANG_CXX_FLAGS}" CACHE STRING "Additional flags to be passed to libclang when parsing with clang")
set(LIBCLANG_LOCATION "${LIBCLANG_LOCATION}" CACHE STRING "Location of the libclang library")

# Detect Python
find_package(Python)

# subdirs
add_subdirectory(bin)            # Executables
add_subdirectory(cmake)          # Exported Cpp2Py-Config
add_subdirectory(include)
add_subdirectory(lib)

# The python modules
add_subdirectory(cpp2py)
add_subdirectory(cpp2rst)
add_subdirectory(cpp2cxx)

# Configure and install the file to source to setup the environment variables 
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/cpp2pyvars.sh.in ${CMAKE_CURRENT_BINARY_DIR}/cpp2pyvars.sh @ONLY)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/cpp2pyvars.sh DESTINATION  ${CMAKE_INSTALL_PREFIX}/share)

###############################################################
#
#            final_message_for_user
#   Remind the user how to set up his/her variables
#
###############################################################
if (NOT LIBCLANG_LOCATION)
 MESSAGE(STATUS "**************************************************************************************")
 MESSAGE(STATUS "**************************          WARNING           ********************************")
 MESSAGE(STATUS "   ")
 MESSAGE(STATUS "    Can not find libclang                                                           ")
 MESSAGE(STATUS "    You can use cpp2py to compile a code, but c++2py, c++2rst, c++2cxx will not work")
 MESSAGE(STATUS "   ")
 MESSAGE(STATUS "**************************          WARNING           ********************************")
 MESSAGE(STATUS "**************************************************************************************")
endif()

MESSAGE(STATUS "***************************************************************")
MESSAGE(STATUS "* Use :                                                        ")
MESSAGE(STATUS "*   source ${CMAKE_INSTALL_PREFIX}/share/cpp2pyvars.sh         ")
MESSAGE(STATUS "*                                                              ") 
MESSAGE(STATUS "*   to set up the environment variables                        ") 
MESSAGE(STATUS "***************************************************************")

#--------------------------------------------------------
# Process and install modulefile
#--------------------------------------------------------
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/cpp2py.modulefile.in ${CMAKE_CURRENT_BINARY_DIR}/cpp2py.modulefile @ONLY)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/cpp2py.modulefile DESTINATION ${CMAKE_INSTALL_PREFIX}/share)

#--------------------------------------------------------
# Packaging
#--------------------------------------------------------
option(BUILD_DEBIAN_PACKAGE "Build a deb package" OFF)
if(BUILD_DEBIAN_PACKAGE)
  if(NOT CMAKE_INSTALL_PREFIX STREQUAL "/usr")
    message(FATAL_ERROR "CMAKE_INSTALL_PREFIX must be /usr for packaging")
  endif()
  set(CPACK_GENERATOR "DEB")
  set(CPACK_PACKAGE_VERSION ${CPP2PY_VERSION})
  set(CPACK_PACKAGE_CONTACT "https://github.com/TRIQS/cpp2py")
  execute_process(COMMAND dpkg --print-architecture OUTPUT_VARIABLE CPACK_DEBIAN_PACKAGE_ARCHITECTURE OUTPUT_STRIP_TRAILING_WHITESPACE)
  set(CPACK_DEBIAN_PACKAGE_DEPENDS "python, python-mako, python-numpy, python-h5py, python-scipy")
  SET(CPACK_DEBIAN_PACKAGE_SHLIBDEPS ON)
  SET(CPACK_DEBIAN_PACKAGE_GENERATE_SHLIBS ON)
  include(CPack)
endif()
