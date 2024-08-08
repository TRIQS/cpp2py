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
#include "../py_converter.hpp"
#include "../traits.hpp"

namespace cpp2py {

  template <typename T> struct py_converter<std::optional<T>> {

    using conv = py_converter<std::decay_t<T>>;

    template <typename O> static PyObject *c2py(O &&op) {
      static_assert(is_instantiation_of_v<std::optional, std::decay_t<O>>, "Logic Error");
      if (!bool(op)) Py_RETURN_NONE;
      return conv::c2py(*(std::forward<O>(op)));
    }

    static bool is_convertible(PyObject *ob, bool raise_exception) { return ((ob == Py_None) or conv::is_convertible(ob, raise_exception)); }

    static std::optional<T> py2c(PyObject *ob) {
      if (ob == Py_None) return {};
      return std::optional<T>{conv::py2c(ob)};
    }
  };
} // namespace cpp2py
