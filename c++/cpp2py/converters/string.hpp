#pragma once
#include "../pyref.hpp"

namespace cpp2py {

  template <> struct py_converter<std::string> {

    static PyObject *c2py(std::string const &x) { return PyUnicode_FromString(x.c_str()); }

    static std::string py2c(PyObject *ob) { return PyUnicode_AsUTF8(ob); }

    static bool is_convertible(PyObject *ob, bool raise_exception) {
      if (PyUnicode_Check(ob) or PyUnicode_Check(ob)) return true;
      if (raise_exception) { PyErr_SetString(PyExc_TypeError, ("Cannot convert "s + to_string(ob) + " to string"s).c_str()); }
      return false;
    }
  };

  template <> struct py_converter<char> {

    static PyObject *c2py(char c) { return PyUnicode_FromString(&c); }

    static char py2c(PyObject *ob) { return PyUnicode_AsUTF8(ob)[0]; }

    static bool is_convertible(PyObject *ob, bool raise_exception) {
      if (PyUnicode_Check(ob) and PyUnicode_GET_LENGTH(ob) == 1) return true;
      if (raise_exception) { PyErr_SetString(PyExc_TypeError, ("Cannot convert "s + to_string(ob) + " to char"s).c_str()); }
      return false;
    }
  };

  template <> struct py_converter<unsigned char> {

    static PyObject *c2py(unsigned char c) { return PyBytes_FromStringAndSize(reinterpret_cast<char *>(&c), 1); }

    static unsigned char py2c(PyObject *ob) { return static_cast<unsigned char>(PyBytes_AsString(ob)[0]); }

    static bool is_convertible(PyObject *ob, bool raise_exception) {
      if (PyBytes_Check(ob) and PyBytes_Size(ob) == 1) return true;
      if (raise_exception) { PyErr_SetString(PyExc_TypeError, ("Cannot convert "s + to_string(ob) + " to unsigned char"s).c_str()); }
      return false;
    }
  };

  template <> struct py_converter<const char *> {

    static PyObject *c2py(const char *x) { return PyUnicode_FromString(x); }

    static const char *py2c(PyObject *ob) { return PyUnicode_AsUTF8(ob); }

    static bool is_convertible(PyObject *ob, bool raise_exception) {
      if (PyUnicode_Check(ob)) return true;
      if (raise_exception) { PyErr_SetString(PyExc_TypeError, ("Cannot convert "s + to_string(ob) + " to string"s).c_str()); }
      return false;
    }
  };

} // namespace cpp2py
