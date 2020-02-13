#pragma once

#include <Python.h>

#if PY_MAJOR_VERSION >= 3

// Python3 treats all ints as longs, PyInt_X functions have been removed.
#define PyInt_Check PyLong_Check
#define PyInt_AsLong PyLong_AsLong
#define PyInt_FromLong PyLong_FromLong

// Python3 strings are unicode, these defines mimic the Python2 functionality.
#define PyString_Check PyUnicode_Check
#define PyString_FromString PyUnicode_FromString
#define PyString_Size PyUnicode_GET_SIZE

// PyUnicode_AsUTF8 isn't available until Python 3.3
#if (PY_VERSION_HEX < 0x03030000)
#define PyString_AsString _PyUnicode_AsString
#else
#define PyString_AsString PyUnicode_AsUTF8
#endif

#endif // PY_MAJOR_VERSION >= 3

// python < 2.6, Py_TYPE  and PyVarObject_HEAD_INIT are not defined:
#ifndef Py_TYPE
#define Py_TYPE(ob) (((PyObject *)(ob))->ob_type)
#endif
#ifndef PyVarObject_HEAD_INIT
#define PyVarObject_HEAD_INIT(type, size) PyObject_HEAD_INIT(type) size,
#endif

// Set Py_TPFLAGS_CHECKTYPES if not defined:
#ifndef Py_TPFLAGS_CHECKTYPES
#define Py_TPFLAGS_CHECKTYPES 0
#endif
