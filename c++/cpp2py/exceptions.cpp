// Copyright (c) 2017 Commissariat à l'énergie atomique et aux énergies alternatives (CEA)
// Copyright (c) 2017 Centre national de la recherche scientifique (CNRS)
// Copyright (c) 2019-2020 Simons Foundation
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

#include <sstream>
#include "cpp2py/exceptions.hpp"

#ifndef CPP2PY_TRACE_MAX_FRAMES
#define CPP2PY_TRACE_MAX_FRAMES 50
#endif

#ifdef __GNUC__

#include <stdio.h>
#include <stdlib.h>
#include <sys/wait.h>
#include <unistd.h>
#include <execinfo.h>
#include <cxxabi.h>
#include <stdlib.h>
#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <iterator>

namespace cpp2py {

  std::string demangle(const char *name) {
    std::stringstream fs;
    int status;
    char *demangled = abi::__cxa_demangle(name, NULL, NULL, &status);
    if (!status) {
      std::string res(demangled);
      fs << res;
      free(demangled);
    } else
      fs << name;
    return fs.str();
  }

  std::string demangle(std::string const &name) { return demangle(name.c_str()); }

  std::string stack_trace() {
    std::ostringstream buffer;
    void *stack[CPP2PY_TRACE_MAX_FRAMES + 1];
    std::size_t depth = backtrace(stack, CPP2PY_TRACE_MAX_FRAMES + 1);
    if (!depth)
      buffer << "  empty  " << std::endl;
    else {
      char **symbols = backtrace_symbols(stack, depth);
      for (std::size_t i = 1; i < depth; ++i) {
        std::string symbol = symbols[i];
        std::istringstream iss(symbol);
        std::vector<std::string> strs{std::istream_iterator<std::string>{iss}, std::istream_iterator<std::string>{}};
        for (auto const &x : strs) buffer << " " << demangle(x);
        buffer << std::endl;
      }
      free(symbols);
    }
    return buffer.str();
  }

} // namespace cpp2py
#else

namespace cpp2py {
  std::string stack_trace() { return std::string("stacktrace only available in gcc"); }
} // namespace cpp2py

#endif

namespace cpp2py {

  bool exception::show_cpp_trace = false;

  exception::exception() : std::exception() { _trace = stack_trace(); }

  const char *exception::what() const noexcept {
    std::stringstream out;
    out << acc.str() << "\n.. Error occurred ";
    int is_init;
    //MPI_Initialized(&is_init);
    //if (is_init) {
    //  int rank;
    //  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    //  out << " on node " << rank;
    // }
    out << "\n";
    if (show_cpp_trace) out << ".. C++ trace is : " << _trace << "\n";
    _what = out.str();
    return _what.c_str();
  }

} // namespace cpp2py
