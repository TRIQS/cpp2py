// Copyright (c) 2017 Commissariat à l'énergie atomique et aux énergies alternatives (CEA)
// Copyright (c) 2017 Centre national de la recherche scientifique (CNRS)
// Copyright (c) 2020 Simons Foundation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0.txt
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// Authors: Olivier Parcollet, Nils Wentzell

#include "./misc.hpp"

namespace cpp2py {

  // Check python version.
  // The version passed here is the version of libpython used to compile the module
  // as the module includes this file.
  // MUST be in cpp. do NOT put this function in hpp.
  bool check_python_version(const char *module_name, long version_hex, long version_major, long version_minor, long version_micro) {

    std::stringstream out;

    // Check that the python version of Python.h used to:
    //    -  compile the module including c2py.hpp
    //          (arguments of this function and frozen at compile time of the module).
    //    -  compile this file, hence libc2py.
    //          (PY_VERSION_HEX et al. below, determined by the Python.h used to compile this file)
    //  are identical.
    if (version_hex != PY_VERSION_HEX) {
      out << "\n\n  Can not load the c2py module "                    //
          << (module_name ? module_name : "") << " ! \n\n"            //
          << "    The c2py library was compiled with Python version " //
          << std::hex << PY_VERSION_HEX << std::dec                   //
          << "    i.e. " << PY_MAJOR_VERSION << '.' << PY_MINOR_VERSION << '.' << PY_MICRO_VERSION
          << "\n    but the python extension is compiled with Python version " //
          << std::hex << version_hex << std::dec << " i.e. " << version_major << '.' << version_minor << '.' << version_micro
          << "\n    They should be identical.\n";
    }

    // Check that the python version of :
    //    -  the interpreter currently running, picked up from the sys module at runtime.
    //    -  Python.h used to compile the module including c2py.hpp
    //          (arguments of this function and frozen at compile time of the module).
    //  are identical.
    auto sys            = pyref::module("sys");
    auto rt_version_hex = PyLong_AsLong(sys.attr("hexversion"));
    std::cerr << rt_version_hex << " = ? " << version_hex;
    if (rt_version_hex != version_hex) {
      auto rt_version = sys.attr("version_info");
      out << "\n\n  Can not load the c2py module "                                        //
          << (module_name ? module_name : "") << " ! \n\n"                                //
          << "    The c2py library was compiled with Python version "                     //
          << std::hex << version_hex << std::dec                                          //
          << "    i.e. " << version_major << '.' << version_minor << '.' << version_micro //
          << "\n    but the python intepreter has version "                               //
          << std::hex << rt_version_hex << std::dec << " i.e. "                           //
          << PyLong_AsLong(rt_version.attr("major")) << '.'                               //
          << PyLong_AsLong(rt_version.attr("minor")) << '.'                               //
          << PyLong_AsLong(rt_version.attr("micro")) << '.'                               //
          << "\n    They should be identical.\n";
    }

    if (out.str().empty()) return true;
    PyErr_SetString(PyExc_ImportError, out.str().c_str());
    return false;
  }

} // namespace cpp2py
