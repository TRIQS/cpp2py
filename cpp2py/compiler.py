import imp, os, sys, subprocess, hashlib, re, tempfile
import libclang_config as Config

cxx_compiler = Config.CXX_COMPILER

cmakelist = """
cmake_minimum_required(VERSION 2.8)
project(cpp2py_magic CXX)
set(CMAKE_BUILD_TYPE Release)
set(BUILD_SHARED_LIBS ON)
add_compile_options( -std=c++14 )
find_package(Cpp2Py REQUIRED)
include_directories(${CMAKE_SOURCE_DIR})
add_cpp2py_module(ext)
find_package(TRIQS REQUIRED)
target_link_libraries(ext triqs)
"""

def print_out (m, out) : 
   l = (70 - len(m))/2
   print l*'-' + m + l*'-' + '\n' + out 

def execute(command, message):
    #print "EXEC", command
    try:
       out = subprocess.check_output(command, stderr=subprocess.STDOUT, shell=True)
    except subprocess.CalledProcessError as E :
       print_out (message + " error ", E.output)
       raise RuntimeError, "Error"
    #if verbosity>0: 
    #print_out(message, out)
    print message

def compile(code, verbosity =0, only=(), modules = ()):
    """
    Takes the c++ code, call c++2py on it and compile the whole thing into a module.
    """
    use_GIL = False
    #if not GIL, we replace std::cout by py_stream for capture in the notebook
    if not use_GIL :
        code = re.sub("std::cout", "cpp2py::py_stream()", code)

    key = code, sys.version_info, sys.executable
    module_dirname = tempfile.mkdtemp("cpp2py_compiler_" + hashlib.md5(str(key).encode('utf-8')).hexdigest())
    module_name = "ext"
    module_path = os.path.join(module_dirname, 'ext.so')

    old_cwd = os.getcwd()
    try:
        os.chdir(module_dirname)

        with open('ext.cpp', 'w') as f:
            f.write("#include <cpp2py/py_stream.hpp>\n")
            f.write(code)

        # Call cmake
        with open('CMakeLists.txt', 'w') as f: f.write(cmakelist)
        execute("cmake . -Wno-dev  -DCMAKE_CXX_COMPILER="+ cxx_compiler, "cmake")

        # Call cpp2py
        only_list = ','.join(only)
        only_list = (" --only " + only_list) if only_list else '' 
        execute("c++2py ./ext.cpp --cxxflags='-std=c++14' -p -m ext -o ext "  + ''.join('-C %s'%x for x in modules) + only_list, "c++2py")

        # Call make
        execute ("make -j2  ", "make")
 
        print "Done"

    finally:
        os.chdir(old_cwd)

    module = imp.load_dynamic(module_name, module_path)
    module.workdir = module_dirname
    return module
