configure_file(${CMAKE_CURRENT_SOURCE_DIR}/Cpp2PyConfig.cmake.in ${CMAKE_CURRENT_BINARY_DIR}/Cpp2PyConfig.cmake @ONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/Cpp2PyConfigVersion.cmake.in ${CMAKE_CURRENT_BINARY_DIR}/Cpp2PyConfigVersion.cmake @ONLY)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/Cpp2PyConfig.cmake DESTINATION  ${CMAKE_INSTALL_PREFIX}/share/cmake)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/Cpp2PyConfigVersion.cmake DESTINATION  ${CMAKE_INSTALL_PREFIX}/share/cmake)

