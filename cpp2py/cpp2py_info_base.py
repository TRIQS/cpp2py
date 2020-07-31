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

import re


def find_in_table(x, table):
    r = set()
    for a,b in list(table.items()):
        if re.compile(a).search(x): r.add(b)
    return r


def get_converters(info_cls, can):
    return find_in_table(can, info_cls.table_converters)


def get_imports(info_cls, can):
    return find_in_table(can, info_cls.table_imports)


class Cpp2pyInfoStd:

    # No Python module to import for std:: object
    table_imports = {}

    # Which converters to import : syntax is :
    # regex to match the canonical C++ type : full name of the converters
    _table_converters = {
        'std::.*complex' : 'complex',
        'std::.*map' : 'map',
        'std::.*set' : 'set',
        'std::.*vector' : 'vector',
        'std::.*string' : 'string',
        'std::.*function' : 'function',
        'std::.*pair' : 'pair',
        'std::.*tuple' : 'tuple',
        'std::.*optional' : 'optional',
        'std::.*variant' : 'variant',
        'std::.*array' : 'std_array',
        }

    table_converters = dict ( (k, "cpp2py/converters/%s.hpp"%v) for (k,v) in list(_table_converters.items()))

