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
#include "../pyref.hpp"
#include "./complex.hpp"

#include <cstddef>

#include <numpy/arrayobject.h>

namespace cpp2py {

  // PyObject *
  template <> struct py_converter<PyObject *> {
    static PyObject *c2py(PyObject *ob) { return ob; }
    static PyObject *py2c(PyObject *ob) { return ob; }
    static bool is_convertible(PyObject *ob, bool raise_exception) { return true; }
  };

  // --- bool
  template <> struct py_converter<bool> {
    static PyObject *c2py(bool b) {
      if (b)
        Py_RETURN_TRUE;
      else
        Py_RETURN_FALSE;
    }
    static bool py2c(PyObject *ob) { return ob == Py_True; }
    static bool is_convertible(PyObject *ob, bool raise_exception) {
      if (PyBool_Check(ob)) return true;
      if (raise_exception) { PyErr_SetString(PyExc_TypeError, ("Cannot convert "s + to_string(ob) + " to bool"s).c_str()); }
      return false;
    }
  };

  // --- long

  namespace details {
    template <typename I> struct py_converter_impl {
      static PyObject *c2py(I i) { return PyLong_FromLong(long(i)); }
      static I py2c(PyObject *ob) {
        if (PyLong_Check(ob)) { return I(PyLong_AsLong(ob)); }
        // Convert NPY Scalar Type to Builtin Type
        pyref py_builtin = PyObject_CallMethod(ob, "item", NULL);
        return I(PyLong_AsLong(py_builtin));
      }
      static bool is_convertible(PyObject *ob, bool raise_exception) {
        _import_array();
        if (PyLong_Check(ob)) return true;
        if (PyArray_CheckScalar(ob)) {
          pyref py_arr = PyArray_FromScalar(ob, NULL);
          if (PyArray_ISINTEGER((PyObject *)py_arr)) return true;
        }
        if (raise_exception) { PyErr_SetString(PyExc_TypeError, ("Cannot convert "s + to_string(ob) + " to integer type"s).c_str()); }
        return false;
      }
    };
  } // namespace details

  template <> struct py_converter<long> : details::py_converter_impl<long> {};
  template <> struct py_converter<int> : details::py_converter_impl<int> {};
  template <> struct py_converter<unsigned int> : details::py_converter_impl<unsigned int> {};
  template <> struct py_converter<unsigned long> : details::py_converter_impl<unsigned long> {};
  template <> struct py_converter<unsigned long long> : details::py_converter_impl<unsigned long long> {};


  // --- byte

  template <> struct py_converter<std::byte> {
    static PyObject *c2py(std::byte b) { return PyBytes_FromStringAndSize(reinterpret_cast<char *>(&b), 1); }
    static std::byte py2c(PyObject *ob) { return static_cast<std::byte>(PyBytes_AsString(ob)[0]); }
    static bool is_convertible(PyObject *ob, bool raise_exception) {
      if (PyBytes_Check(ob) and PyBytes_Size(ob) == 1) return true;
      if (raise_exception) { PyErr_SetString(PyExc_TypeError, ("Cannot convert "s + to_string(ob) + " to byte"s).c_str()); }
      return false;
    }
  };

  // --- double

  template <> struct py_converter<double> {
    static PyObject *c2py(double x) { return PyFloat_FromDouble(x); }
    static double py2c(PyObject *ob) {
      if (PyFloat_Check(ob) || PyLong_Check(ob)) { return PyFloat_AsDouble(ob); }
      // Convert NPY Scalar Type to Builtin Type
      pyref py_builtin = PyObject_CallMethod(ob, "item", NULL);
      return PyFloat_AsDouble(py_builtin);
    }
    static bool is_convertible(PyObject *ob, bool raise_exception) {
      _import_array();
      if (PyFloat_Check(ob) || PyLong_Check(ob)) return true;
      if (PyArray_CheckScalar(ob)) {
        pyref py_arr = PyArray_FromScalar(ob, NULL);
        if (PyArray_ISINTEGER((PyObject*)py_arr) or PyArray_ISFLOAT((PyObject*)py_arr)) return true;
      }
      if (raise_exception) { PyErr_SetString(PyExc_TypeError, ("Cannot convert "s + to_string(ob) + " to double"s).c_str()); }
      return false;
    }
  };

} // namespace cpp2py
