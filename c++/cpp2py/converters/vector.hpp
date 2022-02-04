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
#include <vector>
#include <string>
#include <cstddef>
#include <numpy/arrayobject.h>

#include "../traits.hpp"
#include "../pyref.hpp"
#include "../macros.hpp"
#include "../numpy_proxy.hpp"

namespace cpp2py {

  // Convert vector to numpy_proxy, WARNING: Deep Copy
  template <typename V> numpy_proxy make_numpy_proxy_from_vector(V &&v) {
    static_assert(is_instantiation_of_v<std::vector, std::decay_t<V>>, "Logic error");
    using value_type = typename std::remove_reference_t<V>::value_type;

    auto *vec_heap        = new std::vector<value_type>{std::forward<V>(v)};
    auto delete_pycapsule = [](PyObject *capsule) {
      auto *ptr = static_cast<std::vector<value_type> *>(PyCapsule_GetPointer(capsule, "guard"));
      delete ptr;
    };
    PyObject *capsule = PyCapsule_New(vec_heap, "guard", delete_pycapsule);

    return {1, // rank
            npy_type<value_type>,
            (void *)vec_heap->data(),
            std::is_const_v<value_type>,
            std::vector<long>{long(vec_heap->size())}, // extents
            std::vector<long>{sizeof(value_type)},     // strides
            capsule};
  }

  // Make a new vector from numpy view
  template <typename T> std::vector<T> make_vector_from_numpy_proxy(numpy_proxy const &p) {
    EXPECTS(p.extents.size() == 1);
    EXPECTS(p.strides[0] % sizeof(T) == 0);

    long size = p.extents[0];
    long step = p.strides[0] / sizeof(T);

    std::vector<T> v(size);

    T *data = static_cast<T *>(p.data);
    for (long i = 0; i < size; ++i) v[i] = *(data + i * step);

    return v;
  }

  // --- Special Case: vector<bytes> <-> PyBytes

  template <> struct py_converter<std::vector<std::byte>> {

    static PyObject *c2py(std::vector<std::byte> const &v) {
      auto *char_ptr = reinterpret_cast<const char *>(v.data());
      return PyBytes_FromStringAndSize(char_ptr, v.size());
    }

    // --------------------------------------

    static bool is_convertible(PyObject *ob, bool raise_exception) {
      bool is_bytes_ob = PyBytes_Check(ob);
      if (raise_exception and not is_bytes_ob) {
        PyErr_SetString(PyExc_TypeError, ("Cannot convert "s + to_string(ob) + " to std::vector<byte> as it is not a python bytes object"s).c_str());
      }
      return is_bytes_ob;
    }

    // --------------------------------------

    static std::vector<std::byte> py2c(PyObject *ob) {
      auto size    = PyBytes_Size(ob);
      auto *buffer = reinterpret_cast<std::byte *>(PyBytes_AsString(ob));
      return {buffer, buffer + size};
    }
  };

  // --------------------------------------

  template <typename T> struct py_converter<std::vector<T>> {

    template <typename V> static PyObject *c2py(V &&v) {
      static_assert(is_instantiation_of_v<std::vector, std::decay_t<V>>, "Logic error");
      using value_type = typename std::remove_reference_t<V>::value_type;

      if constexpr (has_npy_type<value_type>) {
        return make_numpy_proxy_from_vector(std::forward<V>(v)).to_python();
      } else { // Convert to Python List
        PyObject *list = PyList_New(0);
        for (auto &x : v) {
          pyref y;
          if constexpr (std::is_reference_v<V>) {
            y = py_converter<value_type>::c2py(x);
          } else { // Vector passed as rvalue
            y = py_converter<value_type>::c2py(std::move(x));
          }
          if (y.is_null() or (PyList_Append(list, y) == -1)) {
            Py_DECREF(list);
            return NULL;
          } // error
        }
        return list;
      }
    }

    // --------------------------------------

    static bool is_convertible(PyObject *ob, bool raise_exception) {
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
        if (raise_exception) {
          PyErr_SetString(PyExc_TypeError, ("Cannot convert "s + to_string(ob) + " to std::vector as it is not a sequence"s).c_str());
        }
        return false;
      }

      pyref seq = PySequence_Fast(ob, "expected a sequence");
      int len   = PySequence_Size(ob);
      for (int i = 0; i < len; i++) {
        if (!py_converter<T>::is_convertible(PySequence_Fast_GET_ITEM((PyObject *)seq, i), raise_exception)) { // borrowed ref
          return false;
        }
      }
      return true;
    }

    // --------------------------------------

    static std::vector<T> py2c(PyObject *ob) {
      // Special case: 1-d ndarray of builtin type
      if (PyArray_Check(ob)) {
        PyArrayObject *arr = (PyArrayObject *)(ob);
#ifdef PYTHON_NUMPY_VERSION_LT_17
        int rank = arr->nd;
#else
        int rank = PyArray_NDIM(arr);
#endif
        if (rank == 1) return make_vector_from_numpy_proxy<T>(make_numpy_proxy(ob));
      }

      ASSERT(PySequence_Check(ob));
      std::vector<T> res;
      pyref seq = PySequence_Fast(ob, "expected a sequence");
      int len   = PySequence_Size(ob);
      for (int i = 0; i < len; i++) res.push_back(py_converter<std::decay_t<T>>::py2c(PySequence_Fast_GET_ITEM((PyObject *)seq, i))); //borrowed ref
      return res;
    }
  };

} // namespace cpp2py
