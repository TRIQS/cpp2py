add_library(cpp2py signal_handler.cpp exceptions.cpp numpy_proxy.cpp)
add_library(cpp2py::cpp2py ALIAS cpp2py)

target_compile_options(cpp2py PRIVATE -std=c++14 -fPIC)
target_include_directories(cpp2py
  PUBLIC
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_PREFIX}/include>
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/c++>
  )

# Install the library in lib and export the cpp2py target
install(TARGETS cpp2py EXPORT cpp2py-targets DESTINATION lib)
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} DESTINATION include FILES_MATCHING PATTERN "*.hpp" PATTERN "*.h" PATTERN "*.hxx")

# --- Python ---

target_link_libraries(cpp2py PUBLIC python_and_numpy)

# ---------------

# Install the exported targets
install(EXPORT cpp2py-targets NAMESPACE cpp2py:: DESTINATION lib/cmake/cpp2py)
