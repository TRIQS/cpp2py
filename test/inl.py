# Copyright (c) 2017-2018 Commissariat à l'énergie atomique et aux énergies alternatives (CEA)
# Copyright (c) 2017-2018 Centre national de la recherche scientifique (CNRS)
# Copyright (c) 2018-2020 Simons Foundation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http:#www.apache.org/licenses/LICENSE-2.0.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Authors: Olivier Parcollet, Nils Wentzell, tayral

from cpp2py.compiler import compile

code = """
   #include <triqs/gfs.hpp>
   
  int f (int x) { return x+3;}
   
  using namespace triqs::gfs;

  gf<imfreq> ma(int n) { return {{10,Fermion, n}, {2,2}};}
"""

M = compile(code, modules = "triqs", cxxflags = ' -O2 ', no_clean = True)


print(M.f(2))
g = M.ma(100)


if 1: 
    # 
    code = """        
    #include <vector>

    std::vector<int> f(int x) { return {x,x+1};}
    //std::vector<double> g(int x) { return {1.0*x,x+2.0};}
    """

    m= compile(code, verbosity =3, recompile = False)

    assert m.f(1) == [1, 2]

