import os, re, sys, itertools
from mako.template import Template
import cpp2py.clang_parser as CL
import util

def make_converter_cxx(c):
    """
    Parameters
    ------------
     
     c  : AST node for a class

     Return
     --------
     string : The C++ code for the Python <-> C++ converters
    """

    c_members = list(CL.get_members(c, True)) # True : with inherited

    def name_lmax(member_list) : 
        return max(len(m.spelling) for m in member_list)

    def name_format(name) : 
        f =  '{:<%s}'%(name_lmax(c_members)+2)
        return f.format(name)

    def name_format_q(name) : return name_format('"%s"'%name) 
 
    def type_format(name) : 
        f  = '{:<%s}'%(max(len(m.type.spelling) for m in c_members))
        return f.format(name)

    # Mako render
    tpl = Template(filename= util.script_path() + '/mako/converters.hxx', strict_undefined=True)
    rendered = tpl.render(c= c, CL = CL, c_members = c_members, name_lmax = name_lmax, name_format = name_format, name_format_q = name_format_q, type_format = type_format)
    return util.clean_end_and_while_char(rendered)

