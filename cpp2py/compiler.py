# Copyright (c) 2017-2018 Commissariat à l'énergie atomique et aux énergies alternatives (CEA)
# Copyright (c) 2017-2018 Centre national de la recherche scientifique (CNRS)
# Copyright (c) 2018-2020 Simons Foundation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http:#www.apache.org/licenses/LICENSE-2.0.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Authors: Gregory Kramida, Olivier Parcollet, Nils Wentzell, tayral

import imp, os, sys, shutil, subprocess, hashlib, re, tempfile
import cpp2py.libclang_config as Config

cxx_compiler = Config.CXX_COMPILER

def print_out (m, out) :
   l = (70 - len(m))//2
   print(l*'-' + m + l*'-' + '\n' + out)

def execute(command, message):
    try:
       out = subprocess.check_output(command, stderr=subprocess.STDOUT, shell=True)
    except subprocess.CalledProcessError as E:
       print_out (message + " error ", E.output.decode('utf8'))
       raise RuntimeError("Error")

def compile(code, verbosity =0, only=(), modules = '', cxxflags= '', moduledir = '/tmp', recompile = False, no_clean = False):
    """
    Takes the c++ code, call c++2py on it and compile the whole thing into a module.
    """
    # Additional Compiler Flags
    if os.getenv('CXXFLAGS'): cxxflags = os.getenv('CXXFLAGS') + cxxflags
    cxxflags = " -std=c++17 " + cxxflags

    modules = modules.strip().split(' ')
    #print(modules)

    use_GIL = False
    #if not GIL, we replace std::cout by py_stream for capture in the notebook
    if not use_GIL :
        code = re.sub("std::cout", "cpp2py::py_stream()", code)

    # Put in a specific namespace __cpp2py_anonymous
    lines = code.split('\n')
    pos = next(n for n, l in enumerate(lines) if l.strip() and not l.strip().startswith('#'))
    code = '\n'.join(lines[:pos] + ['namespace __cpp2py_anonymous {'] + lines[pos:] + ['}'])

    #print(code)
    # key for hash
    key = code, sys.version_info, sys.executable, cxxflags, modules, only
    dir_name = hashlib.md5(str(key).encode('utf-8')).hexdigest().strip()
    module_name = "ext"
    module_dirname = moduledir + '/cpp2py_' + dir_name
    module_path = os.path.join(module_dirname, 'ext.so')

    if not os.path.exists(module_dirname) or recompile:
        try :
            os.mkdir(module_dirname)
        except :
            pass

        old_cwd = os.getcwd()
        try:
            os.chdir(module_dirname)

            with open('ext.cpp', 'w') as f:
                f.write("#include <cpp2py/py_stream.hpp>\n")
                f.write(code)

            # Call cmake
            cmakelist = """
                cmake_minimum_required(VERSION 2.8)
                project(cpp2py_magic CXX)
                set(CMAKE_BUILD_TYPE Release)
                set(BUILD_SHARED_LIBS ON)
                add_compile_options( %s )
                find_package(Cpp2Py REQUIRED)
                include_directories(${CMAKE_SOURCE_DIR})
                add_cpp2py_module(ext)
                find_package(TRIQS REQUIRED)
                target_link_libraries(ext triqs)
            """%cxxflags

            with open('CMakeLists.txt', 'w') as f: f.write(cmakelist)
            execute("cmake . -Wno-dev  -DCMAKE_CXX_COMPILER="+ cxx_compiler, "cmake")

            # Call cpp2py
            only_list = ','.join(only)
            only_list = (" --only " + only_list) if only_list else ''
            execute("c++2py ./ext.cpp --cxxflags='" + cxxflags + "' -p -m ext -N __cpp2py_anonymous -o ext "  + ''.join('-C %s'%x for x in modules if x) + only_list, "c++2py")

            # Call make
            execute ("make -j2  ", "make")

            #print("Done")
        except: # we clean if fail
            os.chdir(old_cwd)
            if not no_clean : shutil.rmtree(module_dirname)
            raise

        finally:
            os.chdir(old_cwd)

    module = imp.load_dynamic(module_name, module_path)
    module.workdir = module_dirname
    return module
