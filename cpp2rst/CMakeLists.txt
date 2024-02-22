set(DEST ${CPP2PY_PYTHON_LIB_DEST_ROOT})

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} DESTINATION ${DEST} FILES_MATCHING PATTERN "*.py")

# Link all files with relevant extensions into binary tree
# This allows us to get a working cpp2py Setup already after the build stage
execute_process(COMMAND ${CPP2PY_PYTHON_EXECUTABLE} ${PROJECT_SOURCE_DIR}/bin/cp_rs.py ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_BINARY_DIR} "py")
