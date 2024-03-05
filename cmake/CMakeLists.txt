configure_file(Cpp2PyConfig.cmake.in Cpp2PyConfig.cmake @ONLY)
configure_file(Cpp2PyConfigVersion.cmake.in Cpp2PyConfigVersion.cmake @ONLY)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/Cpp2PyConfig.cmake DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/Cpp2Py)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/Cpp2PyConfigVersion.cmake DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/Cpp2Py)
