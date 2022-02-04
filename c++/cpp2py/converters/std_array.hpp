// Copyright (c) 2018 Commissariat à l'énergie atomique et aux énergies alternatives (CEA)
// Copyright (c) 2018 Centre national de la recherche scientifique (CNRS)
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
#include <array>
#include <numpy/arrayobject.h>

#include "../pyref.hpp"
#include "../numpy_proxy.hpp"
#include "../py_converter.hpp"
#include "./vector.hpp"

namespace cpp2py {

  template <typename T, size_t R> numpy_proxy make_numpy_proxy_from_heap_array(std::array<T, R> *arr_heap) {

    auto delete_pycapsule = [](PyObject *capsule) {
      auto *ptr = static_cast<std::array<T, R> *>(PyCapsule_GetPointer(capsule, "guard"));
      delete ptr;
    };
    PyObject *capsule = PyCapsule_New(arr_heap, "guard", delete_pycapsule);

    return numpy_proxy{1, // rank
                       npy_type<std::decay_t<T>>,
                       (void *)arr_heap->data(),
                       std::is_const_v<T>,
                       std::vector<long>{long(R)},   // extents
                       std::vector<long>{sizeof(T)}, // strides
                       capsule};
  };

  template <typename T, size_t R> numpy_proxy make_numpy_proxy_from_array(std::array<T, R> const &arr) {

    if constexpr (has_npy_type<T>) {
      auto *arr_heap = new std::array<T, R>{arr};
      return make_numpy_proxy_from_heap_array(arr_heap);
    } else {
      auto *arr_heap = new std::array<pyref, R>{};
      std::transform(begin(arr), end(arr), begin(*arr_heap), [](T const &x) { return py_converter<std::decay_t<T>>::c2py(x); });
      return make_numpy_proxy_from_heap_array(arr_heap);
    }
  }

  // Make a new array from numpy view
  template <typename T, size_t R> std::array<T, R> make_array_from_numpy_proxy(numpy_proxy const &p) {
    EXPECTS(p.extents.size() == 1);
    EXPECTS(p.extents[0] == R);

    std::array<T, R> arr;

    if (p.element_type == npy_type<pyref>) {
      auto *data = static_cast<pyref *>(p.data);
      std::transform(data, data + R, begin(arr), [](PyObject *o) { return py_converter<std::decay_t<T>>::py2c(o); });
    } else {
      EXPECTS(p.strides == std::vector<long>{sizeof(T)});
      T *data = static_cast<T *>(p.data);
      std::copy(data, data + R, begin(arr));
    }

    return arr;
  }

  // --------------------------------------

  template <typename T, size_t R> struct py_converter<std::array<T, R>> {

    static PyObject *c2py(std::array<T, R> const &a) { return make_numpy_proxy_from_array(a).to_python(); }

    // --------------------------------------

    static bool is_convertible(PyObject *ob, bool raise_exception) {
      _import_array();

      // Special case: 1-d ndarray of builtin type
      if (PyArray_Check(ob)) {
        PyArrayObject *arr = (PyArrayObject *)(ob);
#ifdef PYTHON_NUMPY_VERSION_LT_17
        int rank = arr->nd;
#else
        int rank = PyArray_NDIM(arr);
#endif
        if (PyArray_TYPE(arr) == npy_type<T> and rank == 1) return true;
      }

      if (!PySequence_Check(ob)) {
        if (raise_exception) { PyErr_SetString(PyExc_TypeError, ("Cannot convert "s + to_string(ob) + " to std::array as it is not a sequence"s).c_str()); }
        return false;
      }

      pyref seq = PySequence_Fast(ob, "expected a sequence");
      int len   = PySequence_Size(ob);
      if (len != R) {
        if (raise_exception) {
          auto s = std::string{"Convertion to std::array<T, R> failed : the length of the sequence ( = "} + std::to_string(len)
             + " does not match R = " + std::to_string(R);
          PyErr_SetString(PyExc_TypeError, s.c_str());
        }
        return false;
      }
      for (int i = 0; i < len; i++) {
        if (!py_converter<std::decay_t<T>>::is_convertible(PySequence_Fast_GET_ITEM((PyObject *)seq, i), raise_exception)) {
          if (PyErr_Occurred()) PyErr_Print();
          return false;
        }
      }
      return true;
    }

    // --------------------------------------

    static std::array<T, R> py2c(PyObject *ob) {
      _import_array();

      // Special case: 1-d ndarray of builtin type
      if (PyArray_Check(ob)) {
        PyArrayObject *arr = (PyArrayObject *)(ob);
#ifdef PYTHON_NUMPY_VERSION_LT_17
        int rank = arr->nd;
#else
        int rank = PyArray_NDIM(arr);
#endif
        if (rank == 1) return make_array_from_numpy_proxy<T, R>(make_numpy_proxy(ob));
      }

      ASSERT(PySequence_Check(ob));
      std::array<T, R> res;
      pyref seq = PySequence_Fast(ob, "expected a sequence");
      for (int i = 0; i < R; i++) res[i] = py_converter<std::decay_t<T>>::py2c(PySequence_Fast_GET_ITEM((PyObject *)seq, i)); // borrowed ref
      return res;
    }
  };

} // namespace cpp2py
