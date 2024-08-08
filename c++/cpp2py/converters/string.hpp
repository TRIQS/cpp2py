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

#pragma once
#include <string>
#include "../py_converter.hpp"

namespace cpp2py {

  template <> struct py_converter<std::string> {

    static PyObject *c2py(std::string const &x) { return PyUnicode_FromString(x.c_str()); }

    static std::string py2c(PyObject *ob) { return PyUnicode_AsUTF8(ob); }

    static bool is_convertible(PyObject *ob, bool raise_exception) {
      if (PyUnicode_Check(ob) or PyUnicode_Check(ob)) return true;
      if (raise_exception) { PyErr_SetString(PyExc_TypeError, ("Cannot convert "s + to_string(ob) + " to string"s).c_str()); }
      return false;
    }
  };

  template <> struct py_converter<char> {

    static PyObject *c2py(char c) { return PyUnicode_FromString(&c); }

    static char py2c(PyObject *ob) { return PyUnicode_AsUTF8(ob)[0]; }

    static bool is_convertible(PyObject *ob, bool raise_exception) {
      if (PyUnicode_Check(ob) and PyUnicode_GET_LENGTH(ob) == 1) return true;
      if (raise_exception) { PyErr_SetString(PyExc_TypeError, ("Cannot convert "s + to_string(ob) + " to char"s).c_str()); }
      return false;
    }
  };

  template <> struct py_converter<unsigned char> {

    static PyObject *c2py(unsigned char c) { return PyBytes_FromStringAndSize(reinterpret_cast<char *>(&c), 1); }

    static unsigned char py2c(PyObject *ob) { return static_cast<unsigned char>(PyBytes_AsString(ob)[0]); }

    static bool is_convertible(PyObject *ob, bool raise_exception) {
      if (PyBytes_Check(ob) and PyBytes_Size(ob) == 1) return true;
      if (raise_exception) { PyErr_SetString(PyExc_TypeError, ("Cannot convert "s + to_string(ob) + " to unsigned char"s).c_str()); }
      return false;
    }
  };

  template <> struct py_converter<const char *> {

    static PyObject *c2py(const char *x) { return PyUnicode_FromString(x); }

    static const char *py2c(PyObject *ob) { return PyUnicode_AsUTF8(ob); }

    static bool is_convertible(PyObject *ob, bool raise_exception) {
      if (PyUnicode_Check(ob)) return true;
      if (raise_exception) { PyErr_SetString(PyExc_TypeError, ("Cannot convert "s + to_string(ob) + " to string"s).c_str()); }
      return false;
    }
  };

} // namespace cpp2py
