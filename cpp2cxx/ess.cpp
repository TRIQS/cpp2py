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


#include <string>
#include <tuple>

namespace concept {
class C {
 public:
 double attempt();
 double accept();
 void reject();
};
}

namespace CTINT { 

struct A {

 /// doc of A
 int i = 3;

 double x = 2;
 double r = 2;
 double t = 2;
 double g = 2;
 double h = 2;
 double ru = 2;
 double xxx = 2;

 double zzzz;
};

template <typename T> struct B {

 /// doc of A
 int ik = 3;

 std::string x;

 T y;
};

}
