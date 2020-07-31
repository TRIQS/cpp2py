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
#include "../traits.hpp"
#include "../pyref.hpp"

namespace cpp2py {

  template <typename T1, typename T2> struct py_converter<std::pair<T1, T2>> {

    template <typename P> static PyObject *c2py(P &&p) {
      static_assert(is_instantiation_of_v<std::pair, std::decay_t<P>>, "Logic error");
      pyref x1 = convert_to_python(std::get<0>(std::forward<P>(p)));
      pyref x2 = convert_to_python(std::get<1>(std::forward<P>(p)));

      if (x1.is_null() or x2.is_null()) return NULL;
      return PyTuple_Pack(2, (PyObject *)x1, (PyObject *)x2);
    }

    static bool is_convertible(PyObject *ob, bool raise_exception) {
      if (!PySequence_Check(ob)) goto _false;
      {
        pyref seq = PySequence_Fast(ob, "expected a sequence");
        if (!py_converter<T1>::is_convertible(PySequence_Fast_GET_ITEM((PyObject *)seq, 0), raise_exception)) goto _false; // borrowed ref
        if (!py_converter<T2>::is_convertible(PySequence_Fast_GET_ITEM((PyObject *)seq, 1), raise_exception)) goto _false; // borrowed ref
        return true;
      }
    _false:
      if (raise_exception) { PyErr_SetString(PyExc_TypeError, ("Cannot convert "s + to_string(ob) + " to std::pair"s).c_str()); }
      return false;
    }

    static std::pair<T1, T2> py2c(PyObject *ob) {
      pyref seq = PySequence_Fast(ob, "expected a sequence");
      return std::make_pair(py_converter<T1>::py2c(PySequence_Fast_GET_ITEM((PyObject *)seq, 0)),
                            py_converter<T2>::py2c(PySequence_Fast_GET_ITEM((PyObject *)seq, 1)));
    }
  };
} // namespace cpp2py
