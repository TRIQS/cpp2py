import imp,os,sys,subprocess, hashlib,re

__version__ = '0.2.0'

triqs_path = "@CMAKE_INSTALL_PREFIX@"
cxx_compiler = "@CMAKE_CXX_COMPILER@"

generator_path = triqs_path + "/share/triqs/cpp2py"
cpp2py_bin = triqs_path + "/bin/c++2py.py"

cmakelist = """
list(APPEND CMAKE_MODULE_PATH %s/share/triqs/cmake)
cmake_minimum_required(VERSION 2.8)
project(triqs_magic CXX)
set(CMAKE_BUILD_TYPE Release)
option(BUILD_SHARED_LIBS "Build shared libraries" ON)
find_package(TRIQS REQUIRED)
add_definitions(${TRIQS_CXX_DEFINITIONS})
include_directories(${CMAKE_SOURCE_DIR} ${TRIQS_INCLUDE_ALL})
add_library(ext MODULE ext_wrap.cpp)
#add_library(ext MODULE ext.cpp ext_wrap.cpp)
set_target_properties(ext PROPERTIES PREFIX "") #eliminate the lib in front of the module name
target_link_libraries(ext ${TRIQS_LIBRARY_ALL})
triqs_set_rpath_for_target(ext)
"""%triqs_path

def make_desc_and_compile(code, verbosity, only):
    """
    Takes the c++ code, call c++2py on it and compile the whole thing into a module.
    """
    use_GIL = False
    #if not GIL, we replace std::cout by triqs::py_out for capture in the notebook
    if not use_GIL :
        code = re.sub("std::cout", "triqs::py_stream()", code)

    key = code, line, sys.version_info, sys.executable

    module_name = "ext"
    module_dirname = os.path.join(self._lib_dir, "_triqs_magic_" + hashlib.md5(str(key).encode('utf-8')).hexdigest())

    if verbosity > 0:
        print "Workdir = ", module_dirname

    module_path = os.path.join(module_dirname, 'ext.so')
    try :
        os.mkdir(module_dirname)
    except :
        pass

    def print_out (m, out) : 
       l = (70 - len(m))/2
       print l*'-' + m + l*'-' + '\n' + out 

    old_cwd = os.getcwd()
    try:
        os.chdir(module_dirname)

        with open('CMakeLists.txt', 'w') as f:
            f.write(cmakelist)

        with open('ext.cpp', 'w') as f:
            f.write("""
#include <triqs/utility/cpp2py_macros.hpp>
#include <triqs/python_tools/py_stream.hpp>
            """)
            f.write(code)

        def execute(command, message):
            try:
               out = subprocess.check_output(command, stderr=subprocess.STDOUT, shell=True)
            except subprocess.CalledProcessError as E :
               print_out ( message + " error ", E.output)
               raise RuntimeError, "Error"
            if verbosity>0: 
               print_out(message, out)

        # Call cpp2py
        only_list = ','.join(only)
        only_list = (" --only " + only_list) if only_list else '' 
        execute(cpp2py_bin + " ./ext.cpp -p -m ext -o ext" + only_list, "c++2py")

        # Call the wrapper generator
        execute (""" PYTHONPATH={generator_path} python ext_desc.py {generator_path}/mako/xxx_wrap.cpp """.format(**globals()), "Wrapper generator")

        # Call cmake
        execute("cmake . -Wno-dev  -DCMAKE_CXX_COMPILER="+ cxx_compiler+ " -DTRIQS_PATH=" + triqs_path, "cmake")

        # Call make
        execute ("make -j2", "make")

    finally:
        os.chdir(old_cwd)

    self._code_cache[key] = module_path
    module = imp.load_dynamic(module_name, module_path)
    # import all object and function in the main namespace
    imported = []
    for k, v in module.__dict__.items():
        if not k.startswith('_'):
            self.shell.push({k: v})
            imported.append(k)
    if verbosity > 0 and imported:
        print_out("Success", "The following objects are ready to use: %s" % ", ".join(imported))

