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
        if (PyArray_ISINTEGER((PyObject*)py_arr) or PyArray_ISFLOAT((PyObject*)py_arr) or PyArray_ISCOMPLEX((PyObject*)py_arr)) return true;
      }
      if (raise_exception) { PyErr_SetString(PyExc_TypeError, ("Cannot convert "s + to_string(ob) + " to complex"s).c_str()); }
      return false;
    }
  };

} // namespace cpp2py
