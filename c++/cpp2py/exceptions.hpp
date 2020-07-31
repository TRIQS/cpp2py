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
#include <exception>
#include <string>
#include <sstream>

#define CPP2PY_ERROR(CLASS, NAME) throw CLASS() << ".. Triqs " << NAME << " at " << __FILE__ << " : " << __LINE__ << "\n\n"
#define CPP2PY_RUNTIME_ERROR CPP2PY_ERROR(cpp2py::runtime_error, "runtime error")
#define CPP2PY_KEYBOARD_INTERRUPT CPP2PY_ERROR(cpp2py::keyboard_interrupt, "Ctrl-C")
#define CPP2PY_ASSERT(X)                                                                                                                             \
  if (!(X)) CPP2PY_RUNTIME_ERROR << BOOST_PP_STRINGIZE(X)

namespace cpp2py {

  /**
   *  Exception with a << stream operator for simple error message.
   */
  class exception : public std::exception {
    std::stringstream acc;
    std::string _trace;
    mutable std::string _what;

    public:
    exception();
    exception(exception const& e) throw() : acc(e.acc.str()), _trace(e._trace), _what(e._what) {}
    exception(exception &&e)      = default;
    virtual ~exception() {}

    static bool show_cpp_trace;

    template <typename T> exception &operator<<(T const &x) { return acc << x, *this; }

    // ???
    //exception &operator<<(const char *mess) {
    //  (*this) << std::string(mess);
    //  return *this;
    //} // to limit code size

    virtual const char *what() const noexcept;
    //virtual const char *trace() const { return _trace.c_str(); }
  };

  class runtime_error : public exception {
    public:
    runtime_error() : exception() {}

    virtual ~runtime_error() {}

    template <typename T> runtime_error &operator<<(T &&x) {
      exception::operator<<(x);
      return *this;
    }
  };

  class keyboard_interrupt : public exception {
    public:
    keyboard_interrupt() : exception() {}

    virtual ~keyboard_interrupt() {}

    template <typename T> keyboard_interrupt &operator<<(T &&x) {
      exception::operator<<(x);
      return *this;
    }
  };

} // namespace cpp2py
