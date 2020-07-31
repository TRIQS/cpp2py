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
#include <type_traits>

namespace cpp2py {

  template <template <typename...> class TMPLT, typename T> struct is_instantiation_of : std::false_type {};
  template <template <typename...> class TMPLT, typename... U> struct is_instantiation_of<TMPLT, TMPLT<U...>> : std::true_type {};
  template <template <typename...> class gf, typename T>
  inline constexpr bool is_instantiation_of_v = is_instantiation_of<gf, std::decay_t<T>>::value;

} // namespace cpp2py
