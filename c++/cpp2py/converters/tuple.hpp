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
#include <array>
#include <tuple>
#include "../py_converter.hpp"
#include "../traits.hpp"

namespace cpp2py {

  template <typename... Types> struct py_converter<std::tuple<Types...>> {

    private:
    using tuple_t = std::tuple<Types...>;

    // c2py implementation
    template <typename T, auto... Is> static PyObject *c2py_impl(T &&t, std::index_sequence<Is...>) {
      auto objs        = std::array<pyref, sizeof...(Is)>{convert_to_python(std::get<Is>(std::forward<T>(t)))...};
      bool one_is_null = std::any_of(std::begin(objs), std::end(objs), [](PyObject *a) { return a == NULL; });
      if (one_is_null) return NULL;
      return PyTuple_Pack(sizeof...(Types), (PyObject *)(objs[Is])...);
    }

    public:
    template <typename T> static PyObject *c2py(T &&t) {
      static_assert(is_instantiation_of_v<std::tuple, std::decay_t<T>>, "Logic Error");
      return c2py_impl(std::forward<T>(t), std::make_index_sequence<sizeof...(Types)>());
    }

    // -----------------------------------------

    private:
    // Helper function needed due to clang v6 and v7 parameter pack issue
    static auto py_seq_fast_get_item(PyObject *seq, Py_ssize_t i) { return PySequence_Fast_GET_ITEM(seq, i); }

    // is_convertible implementation
    template <auto... Is> static bool is_convertible_impl(PyObject *seq, bool raise_exception, std::index_sequence<Is...>) {
      return (py_converter<std::decay_t<Types>>::is_convertible(py_seq_fast_get_item(seq, Is), raise_exception) and ...);
    }

    public:
    static bool is_convertible(PyObject *ob, bool raise_exception) {
      if (not PySequence_Check(ob)) {
        if (raise_exception) {
          PyErr_SetString(PyExc_TypeError, ("Cannot convert "s + to_string(ob) + " to std::tuple as it is not a sequence"s).c_str());
        }
        return false;
      }
      pyref seq = PySequence_Fast(ob, "expected a sequence");
      // Sizes must match! Could we relax this condition to '<'?
      if (PySequence_Fast_GET_SIZE((PyObject *)seq) != std::tuple_size<tuple_t>::value) {
        if (raise_exception) {
          PyErr_SetString(PyExc_TypeError, ("Cannot convert "s + to_string(ob) + " to std::tuple due to improper length"s).c_str());
        }
        return false;
      }
      return is_convertible_impl((PyObject *)seq, raise_exception, std::make_index_sequence<sizeof...(Types)>());
    }

    // -----------------------------------------

    private:
    template <auto... Is> static auto py2c_impl(PyObject *seq, std::index_sequence<Is...>) {
      return std::make_tuple(py_converter<std::decay_t<Types>>::py2c(py_seq_fast_get_item(seq, Is))...);
    }

    public:
    static tuple_t py2c(PyObject *ob) {
      pyref seq = PySequence_Fast(ob, "expected a sequence");
      return py2c_impl((PyObject *)seq, std::make_index_sequence<sizeof...(Types)>());
    }
  };
} // namespace cpp2py
