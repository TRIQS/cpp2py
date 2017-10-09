
set(DEST ${PYTHON_LIB_DEST_ROOT}/cpp2rst)

install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/__init__.py        DESTINATION  ${DEST})
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/cpp2rst.py         DESTINATION  ${DEST})
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/doc.py             DESTINATION  ${DEST})
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/example.py           DESTINATION  ${DEST})
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/render_fnt.py           DESTINATION  ${DEST})
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/synopsis.py           DESTINATION  ${DEST})

