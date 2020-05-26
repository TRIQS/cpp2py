#pragma once
#include <set>

namespace cpp2py {

  template <typename K> struct py_converter<std::set<K>> {

    static PyObject *c2py(std::set<K> const &s) {
      PyObject *set = PySet_New(NULL);
      for (auto &x : s) {
        pyref y = py_converter<K>::c2py(x);
        if (y.is_null() or (PySet_Add(set, y) == -1)) {
          Py_DECREF(set);
          return NULL;
        } // error
      }
      return set;
    }

    static bool is_convertible(PyObject *ob, bool raise_exception) {
      if (!PySet_Check(ob)) goto _false;
      {
        pyref keys_it = PyObject_GetIter(ob);
        pyref key;
        while (bool(key = PyIter_Next(keys_it))) {
          if (!py_converter<K>::is_convertible(key, raise_exception)) goto _false; //borrowed ref
        }
        return true;
      }
    _false:
      if (raise_exception) { PyErr_SetString(PyExc_TypeError, "Cannot convert to std::set"); }
      return false;
    }
 
    static std::set<K> py2c(PyObject *ob) {
      std::set<K> res;
      pyref keys_it = PyObject_GetIter(ob);
      pyref key;
      while (bool(key = PyIter_Next(keys_it))) {
        res.insert(py_converter<K>::py2c(key)); //borrowed ref
      }
      return res;
    }
  };
} // namespace cpp2py
