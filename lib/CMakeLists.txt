#message( STATUS "-------- MPI detection -------------")
#find_package(MPI REQUIRED)
#message(STATUS "MPI CXX compiler : ${MPI_CXX_COMPILER}")
#message(STATUS "MPI_COMPILE_FLAGS : ${TRIQS_COMPILE_FLAGS_MPI} ")
#message(STATUS "MPI_CXX_INCLUDE_PATH : ${MPI_CXX_INCLUDE_PATH}")

# RPATH setting
if(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
 set(CMAKE_INSTALL_NAME_DIR "${CMAKE_INSTALL_PREFIX}/lib")
 set(CMAKE_MACOSX_RPATH 1) # new for cmake 3.x
else()
 set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")
 set(CMAKE_SKIP_BUILD_RPATH FALSE)
 set(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE)
 set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
endif()

set(ALLSOURCES signal_handler.cpp exceptions.cpp)

#add_library(cpp2py-util ${ALLSOURCES})
add_library(cpp2py ${ALLSOURCES})
target_compile_options(cpp2py PRIVATE -std=c++14)
target_include_directories(cpp2py SYSTEM PRIVATE ${PROJECT_SOURCE_DIR}/include/cpp2py)
target_include_directories(cpp2py SYSTEM INTERFACE ${CMAKE_INSTALL_PREFIX}/include)
#target_include_directories(cpp2py-util SYSTEM PRIVATE ${MPI_CXX_INCLUDE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/../include/)
#target_link_libraries(cpp2py-util ${MPI_CXX_LIBRARIES})

# Install the library in lib and prepare an exported cmake file to reimport it
install(TARGETS cpp2py DESTINATION lib EXPORT cpp2py-targets)
install(EXPORT cpp2py-targets DESTINATION share/cmake)
