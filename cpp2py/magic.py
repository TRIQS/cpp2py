"""
=====================
Cpp2py magic
=====================

{CPP2PY_DOC}

"""
from IPython.core.error import UsageError
from IPython.core.magic import Magics, magics_class, line_magic, cell_magic
from IPython.core import display, magic_arguments
from IPython.utils import py3compat
from IPython.utils.io import capture_output
from IPython.paths import get_ipython_cache_dir

__version__ = '0.3.0'

from onfly import make_desc_and_compile

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
    @magic_arguments.argument(
            "-v", "--verbosity", action="count", default=0,
            help="increase output verbosity"
        )
    @magic_arguments.argument(
            '-o', "--only", action='append', default=[],
            help="""Which object to wrap"""
        )

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
        make_desc_and_compile(code, args.verbosity, args.only)

__doc__ = __doc__.format(CPP2PY_DOC=' ' * 8 + Cpp2pyMagics.cpp2py.__doc__)

def load_ipython_extension(ip):
    """Load the extension in IPython."""
    ip.register_magics(Cpp2pyMagics)

