// Copyright (c) 2018 Commissariat à l'énergie atomique et aux énergies alternatives (CEA)
// Copyright (c) 2018 Centre national de la recherche scientifique (CNRS)
// Copyright (c) 2018-2020 Simons Foundation
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
#include <span>
#include <cstddef>
#include "../pyref.hpp"

namespace cpp2py {

  template <> struct py_converter<std::span<std::byte>> {
    // --------------------------------------

    static bool is_convertible(PyObject *ob, bool raise_exception) {
      bool is_bytes_ob = PyBytes_Check(ob);
      if (raise_exception and not is_bytes_ob) {
        PyErr_SetString(PyExc_TypeError, ("Cannot convert "s + to_string(ob) + " to std::span<byte> as it is not a python bytes object"s).c_str());
      }
      return is_bytes_ob;
    }

    // --------------------------------------

    static std::span<std::byte> py2c(PyObject *ob) {
      auto size    = PyBytes_Size(ob);
      auto *buffer = reinterpret_cast<std::byte *>(PyBytes_AsString(ob));
      return {buffer, size_t(size)};
    }
  };

} // namespace cpp2py
