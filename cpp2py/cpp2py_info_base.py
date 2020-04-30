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
        }

    table_converters = dict ( (k, "cpp2py/converters/%s.hpp"%v) for (k,v) in list(_table_converters.items()))

