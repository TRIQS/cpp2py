#pragma once
#include <vector>
#include <numpy/arrayobject.h>

#include "../numpy_proxy.hpp"

namespace cpp2py {

  template <typename T> static void delete_pycapsule(PyObject *capsule) {
    auto *ptr = static_cast<std::unique_ptr<T[]> *>(PyCapsule_GetPointer(capsule, "guard"));
    delete ptr;
  }

  // Convert vector to numpy_proxy, WARNING: Deep Copy
  template <typename T> numpy_proxy make_numpy_proxy_from_vector(std::vector<T> const &v) {

    auto *data_ptr = new std::unique_ptr<T[]>{new T[v.size()]};
    std::copy(begin(v), end(v), data_ptr->get());
    auto capsule = PyCapsule_New(data_ptr, "guard", &delete_pycapsule<T>);

    return {1, // rank
            npy_type<std::remove_const_t<T>>,
            (void *)data_ptr->get(),
            std::is_const_v<T>,
            v_t{static_cast<long>(v.size())}, // extents
            v_t{sizeof(T)},                   // strides
            capsule};
  }

  // Make a new vector from numpy view
  template <typename T> std::vector<T> make_vector_from_numpy_proxy(numpy_proxy const &p) {
    //EXPECTS(p.extents.size() == 1);
    //EXPECTS(p.strides == v_t{sizeof(T)});

    T *data   = static_cast<T *>(p.data);
    long size = p.extents[0];

    std::vector<T> v(size);
    std::copy(data, data + size, begin(v));
    return v;
  }

  // --------------------------------------

  template <typename T> struct py_converter<std::vector<T>> {

    static PyObject *c2py(std::vector<T> const &v) {

      if constexpr (has_npy_type<T>) {
        return make_numpy_proxy_from_vector(v).to_python();
      } else { // Convert to Python List
        PyObject *list = PyList_New(0);
        for (auto const &x : v) {
          pyref y = py_converter<T>::c2py(x);
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
      _import_array();

      if (PyArray_Check(ob)) {
        PyArrayObject *arr = (PyArrayObject *)(ob);
        if (PyArray_TYPE(arr) != npy_type<T>) goto _false;
#ifdef PYTHON_NUMPY_VERSION_LT_17
        int rank = arr->nd;
#else
        int rank = PyArray_NDIM(arr);
#endif
        if (rank != 1) goto _false;
        return true;
      } else if (PySequence_Check(ob)) {
        pyref seq = PySequence_Fast(ob, "expected a sequence");
        int len   = PySequence_Size(ob);
        for (int i = 0; i < len; i++)
          if (!py_converter<T>::is_convertible(PySequence_Fast_GET_ITEM((PyObject *)seq, i), raise_exception)) goto _false; //borrowed ref
        return true;
      } else
      _false:
        if (raise_exception) { PyErr_SetString(PyExc_TypeError, "Cannot convert to std::vector"); }
      return false;
    }

    // --------------------------------------

    static std::vector<T> py2c(PyObject *ob) {
      _import_array();

      if (PyArray_Check(ob)) { return make_vector_from_numpy_proxy<T>(make_numpy_proxy(ob)); }
      //ASSERT(PySequence_Check(ob));
      std::vector<T> res;
      pyref seq = PySequence_Fast(ob, "expected a sequence");
      int len   = PySequence_Size(ob);
      for (int i = 0; i < len; i++) res.push_back(py_converter<T>::py2c(PySequence_Fast_GET_ITEM((PyObject *)seq, i))); //borrowed ref
      return res;
    }
  };

} // namespace cpp2py
