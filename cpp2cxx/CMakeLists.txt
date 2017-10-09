
set(DEST ${PYTHON_LIB_DEST_ROOT}/cpp2cxx)

install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/__init__.py                DESTINATION  ${DEST})
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/cpp2cxx.py                DESTINATION  ${DEST})
#install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/mako                   DESTINATION  ${DEST})




