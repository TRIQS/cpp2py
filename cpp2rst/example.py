# Copyright (c) 2018 Commissariat à l'énergie atomique et aux énergies alternatives (CEA)
# Copyright (c) 2018 Centre national de la recherche scientifique (CNRS)
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
# Authors: Gregory Kramida, Nils Wentzell, tayral

import os

def prepare_example(filename, decal):
    """From the filename, prepare the doc1, doc2, before and after the code
       and compute the lineno of the code for inclusion"""
    filename += ".cpp"
    if not os.path.exists(filename) :
        #print("example file %s (in %s) does not exist"%(filename,os.getcwd()))
        return None, None, None, 0, 0
    ls = open(filename).read().strip().split('\n')
    r = [i for i, l in enumerate(ls) if not (re.match(r"^\s*/?\*",l) or re.match("^\s*//",l))]
    s, e = r[0],r[-1]+1
    assert r == list(range(s,e))
    def cls(w) :
        w = re.sub(r"^\s*/?\*\s?/?",'',w)
        w = re.sub(r"^\s*//\s?",'',w)
        return w
    doc1 = '\n'.join(cls(x) for x in ls[0:s])
    code = '\n'.join(decal*' ' + x.strip() for x in ls[s:e])
    doc2 = '\n'.join(cls(x) for x in ls[e:])
    return code, doc1, doc2, s, e
