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
// Authors: Nils Wentzell

// C.f. https://numpy.org/doc/1.21/reference/c-api/array.html#importing-the-api
#define NO_IMPORT_ARRAY
#define PY_ARRAY_UNIQUE_SYMBOL _cpp2py_ARRAY_API

#include "numpy_proxy.hpp"

namespace cpp2py {

  // Make a new view_info
  PyObject *numpy_proxy::to_python() {

#ifdef PYTHON_NUMPY_VERSION_LT_17
    int flags = NPY_BEHAVED & ~NPY_OWNDATA;
#else
    int flags = NPY_ARRAY_BEHAVED & ~NPY_ARRAY_OWNDATA;
#endif
    // make the array read only
    if (is_const) flags &= ~NPY_ARRAY_WRITEABLE;
    PyObject *result =
       PyArray_NewFromDescr(&PyArray_Type, PyArray_DescrFromType(element_type), rank, extents.data(), strides.data(), data, flags, NULL);
    if (not result) return nullptr; // the Python error is set

    if (!PyArray_Check(result)) {
      PyErr_SetString(PyExc_RuntimeError, "The python object is not a numpy array");
      return nullptr;
    }

    PyArrayObject *arr = (PyArrayObject *)(result);
#ifdef PYTHON_NUMPY_VERSION_LT_17
    arr->base = base;
    assert(arr->flags == (arr->flags & ~NPY_OWNDATA));
#else
    int r     = PyArray_SetBaseObject(arr, base);
    //EXPECTS(r == 0);
    //EXPECTS(PyArray_FLAGS(arr) == (PyArray_FLAGS(arr) & ~NPY_ARRAY_OWNDATA));
#endif
    base = nullptr; // ref is stolen by the new object

    return result;
  }

  // ----------------------------------------------------------

  // Extract a view_info from python
  numpy_proxy make_numpy_proxy(PyObject *obj) {

    if (obj == NULL) return {};
    if (not PyArray_Check(obj)) return {};

    numpy_proxy result;

    // extract strides and lengths
    PyArrayObject *arr = (PyArrayObject *)(obj);

#ifdef PYTHON_NUMPY_VERSION_LT_17
    result.rank = arr->nd;
#else
    result.rank = PyArray_NDIM(arr);
#endif

    result.element_type = PyArray_TYPE(arr);
    result.extents.resize(result.rank);
    result.strides.resize(result.rank);
    result.data = PyArray_DATA(arr);
    // base is ignored, stays at nullptr

#ifdef PYTHON_NUMPY_VERSION_LT_17
    for (long i = 0; i < result.rank; ++i) {
      result.extents[i] = arr->dimensions[i];
      result.strides[i] = arr->strides[i];
    }
#else
    for (size_t i = 0; i < result.rank; ++i) {
      result.extents[i] = PyArray_DIMS(arr)[i];
      result.strides[i] = PyArray_STRIDES(arr)[i];
    }
#endif

    return result;
  }

  // ----------------------------------------------------------

  PyObject *make_numpy_copy(PyObject *obj, int rank, long element_type) {

    if (obj == nullptr) return nullptr;

    // From obj, we ask the numpy library to make a numpy, and of the correct type.
    // This handles automatically the cases where :
    //   - we have list, or list of list/tuple
    //   - the numpy type is not the one we want.
    //   - adjust the dimension if needed
    // If obj is an array :
    //   - if Order is same, don't change it
    //   - else impose it (may provoque a copy).
    // if obj is not array :
    //   - Order = FortranOrder or SameOrder - > Fortran order otherwise C

    int flags = 0; //(ForceCast ? NPY_FORCECAST : 0) ;// do NOT force a copy | (make_copy ?  NPY_ENSURECOPY : 0);
                   //if (!(PyArray_Check(obj) ))
    //flags |= ( IndexMapType::traversal_order == indexmaps::mem_layout::c_order(rank) ? NPY_C_CONTIGUOUS : NPY_F_CONTIGUOUS); //impose mem order
#ifdef PYTHON_NUMPY_VERSION_LT_17
    flags |= (NPY_C_CONTIGUOUS); //impose mem order
    flags |= (NPY_ENSURECOPY);
#else
    flags |= (NPY_ARRAY_C_CONTIGUOUS); // impose mem order
    flags |= (NPY_ARRAY_ENSURECOPY);
#endif
    return PyArray_FromAny(obj, PyArray_DescrFromType(element_type), rank, rank, flags, NULL); // new ref
  }

} // namespace nda::python
