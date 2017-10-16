# The desc_file generator from libclang ...
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/libclang_config.py.in ${CMAKE_CURRENT_BINARY_DIR}/libclang_config.py @ONLY)

set(DEST ${PYTHON_LIB_DEST_ROOT}/cpp2py)

install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/__init__.py                DESTINATION  ${DEST})
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/libclang_config.py         DESTINATION  ${DEST})
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/clang_parser.py            DESTINATION  ${DEST})
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/cpp2desc.py                DESTINATION  ${DEST})
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/wrap_generator.py          DESTINATION  ${DEST})
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/dependency_analyzer.py     DESTINATION  ${DEST})
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/converters.py              DESTINATION  ${DEST})
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/cpp2py_info_base.py        DESTINATION  ${DEST})
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/doc.py                     DESTINATION  ${DEST})
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/util.py                    DESTINATION  ${DEST})
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/onfly.py                    DESTINATION  ${DEST})
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/magic.py                    DESTINATION  ${DEST})
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/mako                   DESTINATION  ${DEST})


