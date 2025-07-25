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
#include "./exceptions.hpp"
#include "./py_converter.hpp"
#include <time.h>

// silence warning on intel
#ifndef __INTEL_COMPILER
#pragma clang diagnostic ignored "-Wdeprecated-writable-strings"
#endif
#pragma GCC diagnostic ignored "-Wwrite-strings"

namespace cpp2py {

  inline char *get_current_time() { // helper function to print the time in the CATCH_AND_RETURN macro
    time_t rawtime;
    time(&rawtime);
    return ctime(&rawtime);
  }

// I can use the trace in cpp2py::exception
#define CATCH_AND_RETURN(MESS, RET)                                                                                                                  \
  catch (cpp2py::keyboard_interrupt const &e) {                                                                                                      \
    PyErr_SetString(PyExc_KeyboardInterrupt, e.what());                                                                                              \
    return RET;                                                                                                                                      \
  }                                                                                                                                                  \
  catch (cpp2py::exception const &e) {                                                                                                               \
    auto err = std::string(".. Error occurred at ") + cpp2py::get_current_time() + "\n.. Error " + MESS + "\n.. C++ error was : \n" + e.what();      \
    PyErr_SetString(PyExc_RuntimeError, err.c_str());                                                                                                \
    return RET;                                                                                                                                      \
  }                                                                                                                                                  \
  catch (std::exception const &e) {                                                                                                                  \
    auto err = std::string(".. Error occurred at ") + cpp2py::get_current_time() + "\n.. Error " + MESS + "\n.. C++ error was : \n" + e.what();      \
    PyErr_SetString(PyExc_RuntimeError, err.c_str());                                                                                                \
    return RET;                                                                                                                                      \
  }                                                                                                                                                  \
  catch (...) {                                                                                                                                      \
    auto err = std::string(".. Error occurred at ") + cpp2py::get_current_time() + "\n.. Error " + MESS;                                             \
    PyErr_SetString(PyExc_RuntimeError, err.c_str());                                                                                                \
    return RET;                                                                                                                                      \
  }

  // -----------------------------------
  //    Tools for the implementation of reduce (V2)
  // -----------------------------------

  // auxiliary object to reduce the object into a tuple
  class reductor {
    std::vector<PyObject *> elem;
    PyObject *as_tuple() {
      int l         = elem.size();
      PyObject *tup = PyTuple_New(l);
      for (int pos = 0; pos < l; ++pos) PyTuple_SetItem(tup, pos, elem[pos]);
      return tup;
    }

    public:
    template <typename T> reductor &operator&(T &x) {
      elem.push_back(convert_to_python(x));
      return *this;
    }
    template <typename T> PyObject *apply_to(T &x) {
      x.serialize(*this, 0);
      return as_tuple();
    }
  };

  // inverse : auxiliary object to reconstruct the object from the tuple ...
  class reconstructor {
    PyObject *tup; // borrowed ref
    int pos = 0, pos_max = 0;

    public:
    reconstructor(PyObject *borrowed_ref) : tup(borrowed_ref) { pos_max = PyTuple_Size(tup) - 1; }
    template <typename T> reconstructor &operator&(T &x) {
      if (pos > pos_max) CPP2PY_RUNTIME_ERROR << " Tuple too short in reconstruction";
      x = convert_from_python<T>(PyTuple_GetItem(tup, pos++));
      return *this;
    }
  };

  // no protection for convertion !
  template <typename T> struct py_converter_from_reductor {
    template <typename U> static PyObject *c2py(U &&x) { return reductor{}.apply_to(x); }
    static T py2c(PyObject *ob) {
      T res;
      auto r = reconstructor{ob};
      res.serialize(r, 0);
      return res;
    }
    static bool is_convertible(PyObject *ob, bool raise_exception) { return true; }
  };

  // Check python version.
  // The version passed here is the version of libpython used to compile the module
  // as the module includes this file.
  // MUST be in cpp. do NOT put this function in hpp.
  bool check_python_version(const char *module_name = nullptr, long version_major = PY_MAJOR_VERSION, long version_minor = PY_MINOR_VERSION);

} // namespace cpp2py
