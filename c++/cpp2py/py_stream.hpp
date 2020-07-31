// Copyright (c) 2017-2018 Commissariat à l'énergie atomique et aux énergies alternatives (CEA)
// Copyright (c) 2017-2018 Centre national de la recherche scientifique (CNRS)
// Copyright (c) 2018-2020 Simons Foundation
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

#pragma once
#include <ostream>
#include <sstream>
#include "./pyref.hpp"

namespace cpp2py {

  /**
 */
  class py_stream {

    pyref sys, sys_out;
    void _write(const char *s) { pyref res = PyObject_CallMethod(sys_out, (char*)"write", (char*)"s", s); }

    public:
    py_stream() {
      if (!Py_IsInitialized()) CPP2PY_RUNTIME_ERROR << "Construction of a py_stream before the interpreter is initialized !!";
      sys     = pyref::module("sys");
      sys_out = sys.attr("stdout");
    }

    template <class T> py_stream &operator<<(T const &x) {
      std::stringstream fs;
      fs << x;
      _write(fs.str().c_str());
      return *this;
    }

    // this is the type of std::cout
    typedef std::basic_ostream<char, std::char_traits<char>> CoutType;

    // this is the function signature of std::endl
    typedef CoutType &(*StandardEndLine)(CoutType &);

    // define an operator<< to take in std::endl
    py_stream &operator<<(StandardEndLine manip) {
      _write("\n");
      return *this;
    }
  };

} // namespace cpp2py
