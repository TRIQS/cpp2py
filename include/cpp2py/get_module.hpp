#if (PY_MAJOR_VERSION == 3) and (PY_MINOR_VERSION <= 6)

inline PyObject *PyImport_GetModule(PyObject *name) {

  PyObject *modules = PyImport_GetModuleDict(); /* borrowed */

  if (modules == NULL) {
    PyErr_SetString(PyExc_RuntimeError, "unable to get sys.modules");
    return NULL;
  }

  PyObject *m;
  if (PyDict_CheckExact(modules)) {
    m = PyDict_GetItemWithError(modules, name); /* borrowed */
    Py_XINCREF(m);
  } else {
    m = PyObject_GetItem(modules, name);
    if (m == NULL && PyErr_ExceptionMatches(PyExc_KeyError)) { PyErr_Clear(); }
  }

  return m;
}

#endif
