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

"""
=====================
Cpp2py magic
=====================

{CPP2PY_DOC}

"""
import os
from IPython.core.error import UsageError
from IPython.core.magic import Magics, magics_class, line_magic, cell_magic
from IPython.core import display, magic_arguments
from IPython.utils import py3compat
from IPython.utils.io import capture_output
from IPython.paths import get_ipython_cache_dir

__version__ = '0.3.0'

from .compiler import compile, print_out

@magics_class
class Cpp2pyMagics(Magics):

    def __init__(self, shell):
        super(Cpp2pyMagics, self).__init__(shell=shell)
        self._reloads = {}
        self._code_cache = {}
        self._lib_dir = os.path.join(get_ipython_cache_dir(), 'cpp2py')
        if not os.path.exists(self._lib_dir):
            os.makedirs(self._lib_dir)

    @magic_arguments.magic_arguments()
    @magic_arguments.argument( "-v", "--verbosity", type=int, help="increase output verbosity")
    @magic_arguments.argument( '-o', "--only", action='append', default=[], help="""Which object to wrap""")
    @magic_arguments.argument( '-C', "--converters", action='append', default=[], help="""Modules""")
    @magic_arguments.argument( '--cxxflags', action='append', default = [], help="""Additional compiler flags""")
    @magic_arguments.argument( "--no_clean", action='store_true', default=[], help="""""")
    @cell_magic
    def cpp2py(self, line, cell=None):
        """Compile and import everything from a Cpp2py code cell.

        Takes the c++ code, call c++2py on it and compile the whole thing
        into a module which is then loaded and
        all of its symbols are injected into the user's namespace.

        Usage
        =====
        Prepend ``%%cpp2py`` to your cpp2py code in a cell::

        ``%%cpp2py

        ! put your code here.
        ``
        """

        try:
            # custom saved arguments
            saved_defaults = vars(
                magic_arguments.parse_argstring(self.cpp2py,
                                                self.shell.db['cpp2py']))
            self.cpp2py.parser.set_defaults(**saved_defaults)
        except KeyError:
            saved_defaults = {'verbosity': 0}

        if '-v' in line:
            self.cpp2py.parser.set_defaults(verbosity=0)

        args = magic_arguments.parse_argstring(self.cpp2py, line)
        code = cell if cell.endswith('\n') else cell + '\n'
        cxxflags = ''.join([fl.strip('"').strip("'") for fl in args.cxxflags])
        module = compile(code, verbosity = args.verbosity, only = args.only, cxxflags = cxxflags, modules = ''.join(args.converters), no_clean = args.no_clean)

        # import all object and function in the main namespace
        imported = []
        for k, v in list(module.__dict__.items()):
            if not k.startswith('_'):
                self.shell.push({k: v})
                imported.append(k)
        if args.verbosity and args.verbosity > 0 and imported:
            print_out("Success", "The following objects are ready to use: %s" % ", ".join(imported))

__doc__ = __doc__.format(CPP2PY_DOC=' ' * 8 + Cpp2pyMagics.cpp2py.__doc__)

def load_ipython_extension(ip):
    """Load the extension in IPython."""
    ip.register_magics(Cpp2pyMagics)
