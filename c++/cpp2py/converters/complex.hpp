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

#include <numpy/arrayobject.h>

namespace cpp2py {

  // --- complex

  template <> struct py_converter<std::complex<double>> {
    static PyObject *c2py(std::complex<double> x) { return PyComplex_FromDoubles(x.real(), x.imag()); }
    static std::complex<double> py2c(PyObject *ob) {
      if (PyArray_CheckScalar(ob)) {
        // Convert NPY Scalar Type to Builtin Type
        pyref py_builtin = PyObject_CallMethod(ob, "item", NULL);
        if (PyComplex_Check(py_builtin)) {
          auto r = PyComplex_AsCComplex(py_builtin);
          return {r.real, r.imag};
        } else {
          return PyFloat_AsDouble(py_builtin);
        }
      }

      if (PyComplex_Check(ob)) {
        auto r = PyComplex_AsCComplex(ob);
        return {r.real, r.imag};
      }
      return PyFloat_AsDouble(ob);
    }
    static bool is_convertible(PyObject *ob, bool raise_exception) {
      if (PyComplex_Check(ob) || PyFloat_Check(ob) || PyLong_Check(ob)) return true;
      if (PyArray_CheckScalar(ob)) {
        pyref py_arr = PyArray_FromScalar(ob, NULL);
        if (PyArray_ISINTEGER((PyObject *)py_arr) or PyArray_ISFLOAT((PyObject *)py_arr) or PyArray_ISCOMPLEX((PyObject *)py_arr)) return true;
      }
      if (raise_exception) { PyErr_SetString(PyExc_TypeError, ("Cannot convert "s + to_string(ob) + " to complex"s).c_str()); }
      return false;
    }
  };

} // namespace cpp2py
