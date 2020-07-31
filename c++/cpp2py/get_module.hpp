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

#pragma once

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
