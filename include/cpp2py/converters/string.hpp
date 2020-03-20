#pragma once
//#include <string>

namespace cpp2py {

  template <> struct py_converter<std::string> {

    static PyObject *c2py(std::string const &x) { return PyUnicode_FromString(x.c_str()); }

    static std::string py2c(PyObject *ob) { return PyUnicode_AsUTF8(ob); }

    static bool is_convertible(PyObject *ob, bool raise_exception) {
      if (PyUnicode_Check(ob) or PyUnicode_Check(ob)) return true;
      if (raise_exception) { PyErr_SetString(PyExc_TypeError, "Cannot convert to string"); }
      return false;
    }
  };

  template <> struct py_converter<char> {

    static PyObject *c2py(char c) { return PyUnicode_FromString(&c); }

    static char py2c(PyObject *ob) { return PyUnicode_AsUTF8(ob)[0]; }

    static bool is_convertible(PyObject *ob, bool raise_exception) {
      if (PyUnicode_Check(ob) and PyUnicode_GET_LENGTH(ob) == 1) return true;
      if (raise_exception) { PyErr_SetString(PyExc_TypeError, "Cannot convert to char"); }
      return false;
    }
  };

  template <> struct py_converter<const char *> {

    static PyObject *c2py(const char *x) { return PyUnicode_FromString(x); }

    static const char *py2c(PyObject *ob) { return PyUnicode_AsUTF8(ob); }

    static bool is_convertible(PyObject *ob, bool raise_exception) {
      if (PyUnicode_Check(ob)) return true;
      if (raise_exception) { PyErr_SetString(PyExc_TypeError, "Cannot convert to string"); }
      return false;
    }
  };

} // namespace cpp2py
