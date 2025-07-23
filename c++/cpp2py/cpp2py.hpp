// Copyright (c) 2017-2018 Commissariat à l'énergie atomique et aux énergies alternatives (CEA)
// Copyright (c) 2017-2018 Centre national de la recherche scientifique (CNRS)
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

#define CPP2PY_IGNORE __attribute__((annotate("ignore_in_python")))
#define CPP2PY_ARG_AS_DICT __attribute__((annotate("use_parameter_class")))

#include <Python.h>

#include "./cpp2py/signal_handler.hpp"
#include "./cpp2py/exceptions.hpp"
#include "./cpp2py/pyref.hpp"
#include "./cpp2py/py_converter.hpp"
#include "./cpp2py/misc.hpp"
#include "./cpp2py/converters/basic_types.hpp"

// Remove error on gcc complaining that the specialization
// happens after the first instantiation
#ifdef C2PY_INCLUDED
#include "./cpp2py/converters/std_array.hpp"
#endif

