set(DEST ${PYTHON_LIB_DEST_ROOT})

configure_file(libclang_config.py.in libclang_config.py @ONLY)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/libclang_config.py DESTINATION ${DEST}/cpp2py)

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} DESTINATION ${DEST} FILES_MATCHING PATTERN "*.py" PATTERN "*.cxx" PATTERN "*.rst")

# Link all files with relevant extensions into binary tree
# This allows us to get a working cpp2py Setup already after the build stage
execute_process(COMMAND ${PYTHON_EXECUTABLE} ${PROJECT_SOURCE_DIR}/bin/cp_rs.py ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_BINARY_DIR} "py cxx rst")

###################################################################################
#
#            add_cpp2py_module
#
#   Signagutures :
#		add_cpp2py_module(module_name)
#		add_cpp2py_module(NAME module_name DEPENDS header_dependencies)
#   Arguments :
#		module_name : The name of the module
#		header_dependencies : Header files that the module depends upon
#
#   Set up the compilation of the cpp2py python module
#   Expects a file ${module_name}_desc.py in the CURRENT_SOURCE_DIR
#   Adds a library target ${module_name} which :
#
#	* DEPENDS on MyModule_desc.py and optionally a set of ${header_dependencies}
#	* Builds the wrapper-code ${module_name}_wrap.cxx in CURRENT_BINARY_DIR
#	* Compiles the wrapper-code into the module ${module_name}.so
#
###################################################################################
function(add_cpp2py_module)

  if(${ARGC} EQUAL 1)
    set(module_name ${ARGV0})
  else()
    cmake_parse_arguments(ARG "" "NAME;DIRECTORY" "DEPENDS" ${ARGN})
    set(module_name ${ARG_NAME})
    set(module_dir ${ARG_DIRECTORY})
    set(header_dependencies ${ARG_DEPENDS})
  endif()

  message(STATUS "Adding cpp2py Python module ${module_name}")

  set(desc_name ${CMAKE_CURRENT_SOURCE_DIR}/${module_dir}/${module_name}_desc.py)    # the desc file
  set(wrap_name ${CMAKE_CURRENT_BINARY_DIR}/${module_dir}/${module_name}_wrap.cxx)   # path to the wrapping code

  add_custom_command(OUTPUT ${wrap_name} DEPENDS ${desc_name} ${header_dependencies}
    COMMAND PYTHONPATH=${PROJECT_BINARY_DIR}:${PROJECT_BINARY_DIR}/python:${CPP2PY_BINARY_DIR}:${CPP2PY_ADD_MODULE_ADDITIONAL_PYTHONPATH}:$ENV{PYTHONPATH} ${PYTHON_EXECUTABLE} ${desc_name} ${wrap_name})

  add_library(${module_name} MODULE ${wrap_name})
  set_target_properties(${module_name} PROPERTIES SUFFIX ${PYTHON_MODULE_EXT})
  target_link_libraries(${module_name} cpp2py::cpp2py)

  # We expose only the modules init function in the dynamic symbol table
  # Use only for Linux as OSX Generates various linker warnings
  target_compile_options(${module_name} PRIVATE
    $<$<PLATFORM_ID:Linux>:-fvisibility=hidden>
    $<$<PLATFORM_ID:Linux>:-fvisibility-inlines-hidden>
  )

  set_target_properties(${module_name}
    PROPERTIES
    PREFIX ""  #eliminate the lib in front of the module name
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/${module_dir}
  )

  # Keep a list of every module target.
  # Usage : e.g.  Documentation top target depends on them being built first
  set_property(GLOBAL APPEND PROPERTY CPP2PY_MODULES_LIST ${module_name})

endfunction(add_cpp2py_module)
