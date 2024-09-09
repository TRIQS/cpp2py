#include <array>
#include <complex>
#include <csignal>
#include <cstddef>
#include <iostream> //for std::cout...
#include <string>
using dcomplex = std::complex<double>;

// C.f. https://numpy.org/doc/1.21/reference/c-api/array.html#importing-the-api
#define PY_ARRAY_UNIQUE_SYMBOL _cpp2py_ARRAY_API

// first the basic stuff
#include <cpp2py/cpp2py.hpp>
#include <cpp2py/converters/string.hpp>

// for converters
#include <cpp2py/converters/vector.hpp>
#include <cpp2py/converters/string.hpp>
#include <cpp2py/converters/function.hpp>
#include <algorithm>


//------------------------------------------------------------------------------------------------------
//---------------------   includes and using  -------------------
//------------------------------------------------------------------------------------------------------
#define TRIQS_PYTHON_WRAPPER_MODULE_${module.name}

%for file in module.include_list :
%if file.startswith('<'):
#include ${file}
%elif not file.startswith('_'):
%if file.startswith('~'):
#include "${file[1:]}"
%else:
#include "${file}"
%endif
%endif
%endfor

%for ns in module.using:
using ${ns};
%endfor

<% include_serialization=0 %>
%for c in module.classes.values() :
%if c.serializable == "h5" :
<% include_serialization=1 %>
<% break %>
%endif
%endfor
%if include_serialization==1 :
#include <h5/serialization.hpp>
%endif

using namespace cpp2py;

${module._preamble}


<%
  import re
  def get_c_str_literal(s):
      s = repr(s)[1:-1] # Remove ' ' from repr
      s = re.sub(r"(?<!\\)\"",r"\"", s) # Escape double quotes
      s = re.sub(r"\'",r"'", s) # Unescape single quotes
      return s
%>

//------------------------------------------------------------------------------------------------------
// Second all the classes and enums wrapped by imported modules
// Most converters will be automatically included
//------------------------------------------------------------------------------------------------------

<%
  cpp2py_imported_modules = [m for n, m in sys_modules.items() if hasattr(m,'_get_cpp2py_wrapped_class_enums')]
  cpp2py_imported_modules = dict((m.__name__.split('.')[-1], m) for m in cpp2py_imported_modules).values()
%>

%for M in cpp2py_imported_modules:
 <%
   d = M._get_cpp2py_wrapped_class_enums()
   wrapped_ens = eval(d.get('enums',"()"))
   includes = eval(d['includes'])
 %>

  //--------------------- includes of the imported modules

 %for inc in includes :
  %if inc.startswith('<'):
#include ${inc}
  %else:
#include "${inc}"
  %endif
 %endfor

namespace cpp2py {

//--------------------- Converters of enums --------------------------

%for (c_name_absolute, c_namespace, values) in wrapped_ens:

template <> struct py_converter<${c_name_absolute}> {
 static PyObject * c2py(${c_name_absolute} x) {
   %for n,val in enumerate(values[:-1]) :
    if (x == ${c_namespace}${val}) return PyUnicode_FromString("${val}");
   %endfor
   return PyUnicode_FromString("${values[-1]}"); // last case separate to avoid no return warning of compiler
 }
 static ${c_name_absolute} py2c(PyObject * ob){
   std::string s = PyUnicode_AsUTF8(ob);
   %for n,val in enumerate(values[:-1]) :
    if (s == "${val}") return ${c_namespace}${val};
   %endfor
   return ${c_namespace}${values[-1]};
 }
 static bool is_convertible(PyObject *ob, bool raise_exception) {
   if (!PyUnicode_Check(ob)) {
     if (raise_exception) PyErr_SetString(PyExc_ValueError, "Convertion of C++ enum ${c_name_absolute} : the object is not a string");
     return false;
   }
   std::string s = PyUnicode_AsUTF8(ob);
   %for n,val in enumerate(values) :
    if (s == "${val}") return true;
   %endfor
   if (raise_exception) {
    auto err = "Convertion of C++ enum ${c_name_absolute} : \nThe string \"" + s +"\" is not in [${','.join([str(x) for x in values])}]";
    PyErr_SetString(PyExc_ValueError, err.c_str());
#include <cpp2py/cpp2py.hpp>
   }
   return false;
 }
};

%endfor
## end loop on enums
} // namespace cpp2py
%endfor
## END LOOP ON IMPORTED MODULES
//------------------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------------------

//--------------------- Generated converters --------------------------

%for conv in module.converters.values():
 ${conv.generate()}
%endfor

// We use the order, in the following order (which is necessary for compilation : we need the converters in the implementations)
// - function/method declaration
// - implement type, and all tables
// - implement the converter of types, then of associated regular_type if any.
// - implement functions/method.
//
//----------------------------------------------------------------
//---------------------  function declarations -------------------
//----------------------------------------------------------------

//--------------------- all functions/methods with args, kwds, including constructors -----------------------------

%for py_meth, module_or_class_name, self_c_type in module._all_args_kw_functions() :
 static ${'PyObject*' if not py_meth.is_constructor else 'int'} ${module_or_class_name}_${py_meth.py_name}(PyObject *self, PyObject *args, PyObject *keywds);
%endfor

// ------------------------------- Loop on all classes ----------------------------------------------------

%for c in module.classes.values() :

// ----------------------------
// start class : ${c.py_type}
// ----------------------------
//--------------------- all members  -----------------------------

%for m in c.members:
static PyObject * ${c.py_type}__get_member_${m.py_name} (PyObject *self, void *closure);
%if not m.read_only:
static int ${c.py_type}__set_member_${m.py_name} (PyObject *self, PyObject *value, void *closure);
%endif
%endfor

//--------------------- all properties  -----------------------------

%for p in c.properties :
static PyObject * ${c.py_type}__get_prop_${p.name} (PyObject *self, void *closure);
%if p.setter :
static int ${c.py_type}__set_prop_${p.name} (PyObject *self, PyObject *value, void *closure);
%endif
%endfor

//--------------------- [] -----------------------------

%if "__len__impl" in c.methods :
static Py_ssize_t ${c.py_type}___len__(PyObject *self);
%endif

%if "__getitem__impl" in c.methods :
static PyObject* ${c.py_type}___getitem__(PyObject *self, PyObject *key);
%endif

%if "__setitem__impl" in c.methods :
static int ${c.py_type}___setitem__(PyObject *self, PyObject *key, PyObject *v);
%endif


//--------------------- reduce  -----------------------------

static PyObject* ${c.py_type}___reduce__ (PyObject *self, PyObject *args, PyObject *keywds);
%if c.serializable :
 static PyObject* ${c.py_type}___reduce_reconstructor__ (PyObject *self, PyObject *args, PyObject *keywds);
%endif

//--------------------- hdf5 : write part -----------------------------

static PyObject* ${c.py_type}___write_hdf5__ (PyObject *self, PyObject *args);

//--------------------- repr  -----------------------------

%if c.is_printable :
static PyObject* ${c.py_type}___repr__ (PyObject *self);
static PyObject* ${c.py_type}___str__ (PyObject *self);
%endif

//--------------------- Arithmetic declaration only -----------------------------

% for op_name, op in c.number_protocol.items():
%if op.arity == 2:
static PyObject * ${c.py_type}_${op_name} (PyObject* v, PyObject *w);
%elif op.arity ==1:
static PyObject * ${c.py_type}_${op_name} (PyObject *v);
%endif
%endfor

//--------------------- Comparison  -----------------------------

static PyObject* ${c.py_type}_richcompare (PyObject *a, PyObject *b, int op);

// ----------------------------
// stop class : ${c.py_type}
// ----------------------------
%endfor // cc classes

//----------------------------------------------------------------
//---------------------  implementation --------------------------
//----------------------------------------------------------------

// ------------------------------- Loop on all classes ----------------------------------------------------

%for c in module.classes.values() :
// ----------------------------
// start class : ${c.py_type}
// ----------------------------

//--------------------- define all the types and the converter functions -----------------------------


 // the python type
typedef struct {
    PyObject_HEAD
    ${c.c_type} * _c;
} ${c.py_type};

## The new function, only if there is constructor
%if c.constructor:
// the new
static PyObject* ${c.py_type}_new(PyTypeObject *type, PyObject *args, PyObject *kwds) {
  ${c.py_type} *self;
  self = (${c.py_type} *)type->tp_alloc(type, 0);
  if (self != NULL) self->_c = NULL;
  return (PyObject *)self;
}
%endif

// dealloc
static void ${c.py_type}_dealloc(PyObject* self) {
  auto * c_ptr = ((${c.py_type}*)self)->_c;
  if (c_ptr != NULL) delete c_ptr; // should never be null, but I protect it anyway
  Py_TYPE(self)->tp_free((PyObject*)self);
}

//--------------------- Iterator by wrapping the C++ -----------------------------

%if c.iterator :
 // I used the doc and http://stackoverflow.com/questions/1815812/how-to-create-a-generator-iterator-with-the-python-c-api
 // to write this
 // the iterator type
 typedef struct {
  PyObject_HEAD
  PyObject * container;
  decltype(std::cbegin(std::declval<${c.c_type}>())) iter;
  decltype(std::cend(std::declval<${c.c_type}>())) end;
 } ${c.py_type}__iterator;

 // dealloc
 static void  ${c.py_type}__iterator_dealloc(${c.py_type}__iterator * self) {
   Py_XDECREF(self->container);
   Py_TYPE(self)->tp_free((PyObject*)self);
 }

 // the __iter__ of the iterator type : returns itself
 PyObject* ${c.py_type}__iterator__iter__(PyObject *self) {
  Py_INCREF(self);
  return self;
 }

 // the next() method of the iterator
 PyObject* ${c.py_type}__iterator__iternext__(PyObject *self);

 static PyTypeObject ${c.py_type}__iteratorType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    "${module.name}.${c.py_type}__iterator",            /*tp_name*/
    sizeof(${c.py_type}__iterator),       /*tp_basicsize*/
    0,     /*tp_itemsize*/
    (destructor)${c.py_type}__iterator_dealloc,     /*tp_dealloc*/
    0,     /*tp_print*/
    0,     /*tp_getattr*/
    0,     /*tp_setattr*/
    0,     /*tp_compare*/
    0,     /*tp_repr*/
    0,     /*tp_as_number*/
    0,     /*tp_as_sequence*/
    0,     /*tp_as_mapping*/
    0,     /*tp_hash */
    0,     /*tp_call*/
    0,     /*tp_str*/
    0,     /*tp_getattro*/
    0,     /*tp_setattro*/
    0,     /*tp_as_buffer*/
    Py_TPFLAGS_DEFAULT, /* tp_flags */
    "Internal ${c.py_type} iterator object.",           /* tp_doc */
    0,  /* tp_traverse */
    0,  /* tp_clear */
    0,  /* tp_richcompare */
    0,  /* tp_weaklistoffset */
    ${c.py_type}__iterator__iter__,  /* tp_iter: __iter__() method */
    ${c.py_type}__iterator__iternext__  /* tp_iternext: next() method */
};

// the __iter__ of the main type: return a new iterator. Need to allocate and init it by hand (no init method).
PyObject* ${c.py_type}___iter__(PyObject *self);

%endif

//--------------------- Register as_number  -----------------------------

%if c.number_protocol :
static PyNumberMethods ${c.py_type}_as_number = {

%for op_name in ["add", "subtract", "multiply", "remainder", "divmod", "power", "negative", "positive", "absolute", "bool", "invert", "lshift", "rshift", "and", "xor", "or", "int", "reserved", "float", "inplace_add", "inplace_subtract", "inplace_multiply", "inplace_remainder", "inplace_power", "inplace_lshift", "inplace_rshift", "inplace_and", "inplace_xor", "inplace_or", "floor_divide", "true_divide", "inplace_floor_divide", "inplace_true_divide", "index", "matrix_multiply", "inplace_matrix_multiply"] :
% if op_name in c.number_protocol and c.number_protocol[op_name].arity==2:
 (binaryfunc)${c.py_type}_${op_name}, /*nb_${op_name}*/
% elif op_name in c.number_protocol and c.number_protocol[op_name].arity==1:
 (unaryfunc)${c.py_type}_${op_name}, /*nb_${op_name}*/
% else:
 0, /*nb_${op_name}*/
% endif
%endfor

};
%endif

//--------------------- Register as_mapping  -----------------------------

%if c.has_mapping_protocol :
static PyMappingMethods ${c.py_type}_mapping {
%if "__len__impl" in c.methods :
 (lenfunc)${c.py_type}___len__, /* mp_length */
%else:
NULL, /* mp_length */
%endif
%if "__getitem__impl" in c.methods :
 (binaryfunc)${c.py_type}___getitem__,     /* mp_subscript*/
%else:
NULL,     /* mp_subscript*/
%endif
%if "__setitem__impl" in c.methods :
 (objobjargproc)${c.py_type}___setitem__     /* mp_ass_subscript*/
%else:
NULL     /* mp_ass_subscript*/
%endif
};
%endif

//--------------------- Register members and properties -----------------------------

static PyGetSetDef ${c.py_type}_getseters[] = {

%for mb in [m for m in c.members if not m.read_only] :
{"${mb.py_name}", (getter)${c.py_type}__get_member_${mb.py_name}, ${c.py_type}__set_member_${mb.py_name}, "${get_c_str_literal(mb._generate_doc())}", NULL},
%endfor

%for mb in [m for m in c.members if m.read_only] :
{"${mb.py_name}", (getter)${c.py_type}__get_member_${mb.py_name}, NULL, "${get_c_str_literal(mb._generate_doc())}", NULL},
%endfor

%for p in [p for p in c.properties if p.setter ] :
{"${p.name}", (getter)${c.py_type}__get_prop_${p.name}, ${c.py_type}__set_prop_${p.name}, "${get_c_str_literal(p._generate_doc())}", NULL},
%endfor

%for p in [p for p in c.properties if not p.setter ] :
{"${p.name}", (getter)${c.py_type}__get_prop_${p.name}, NULL, "${get_c_str_literal(p._generate_doc())}", NULL},
%endfor

{NULL}  /* Sentinel */
};

//--------------------- Register the methods for the types  -----------------------------

static PyMethodDef ${c.py_type}_methods[] = {
   %for meth_name, meth in c.methods.items():
    %if not meth_name.startswith('__') :
    {"${meth_name}", (PyCFunction)${c.py_type}_${meth_name}, METH_VARARGS| METH_KEYWORDS ${"|METH_STATIC" if meth.is_static else ""}, "${get_c_str_literal(c.methods[meth_name]._generate_doc())}" },
    %endif
   %endfor
    {"__reduce__", (PyCFunction)${c.py_type}___reduce__, METH_VARARGS, "Internal  " },
    {"__write_hdf5__", (PyCFunction)${c.py_type}___write_hdf5__, METH_VARARGS, "Internal : hdf5 writing via C++ " },
{NULL}  /* Sentinel */
};

//--------------------- The xxxType table  -----------------------------

static PyTypeObject ${c.py_type}Type = {
    PyVarObject_HEAD_INIT(NULL, 0)
    "${module.full_name}.${c.py_type}",/*tp_name*/
    sizeof(${c.py_type}),             /*tp_basicsize*/
    0,                         /*tp_itemsize*/
    (destructor)${c.py_type}_dealloc, /*tp_dealloc*/
    0,                         /*tp_print*/
    0,                         /*tp_getattr*/
    0,                         /*tp_setattr*/
    0,                         /*tp_compare*/
    ${"&%s___repr__"%c.py_type if c.is_printable else 0},           /*tp_repr*/
    ${"&%s_as_number"%c.py_type if c.number_protocol else 0},       /*tp_as_number*/
    0,                         /*tp_as_sequence*/
    ${"&%s_mapping"%c.py_type if c.has_mapping_protocol else 0},    /*tp_as_mapping*/
    0,                         /*tp_hash */
    ${c.py_type+"___call__" if "__call__" in c.methods else 0},                         /*tp_call*/
    ${"&%s___str__"%c.py_type if c.is_printable else 0},                         /*tp_str*/
    0,                         /*tp_getattro*/
    0,                         /*tp_setattro*/
    0,                         /*tp_as_buffer*/
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE, /*tp_flags*/
    "${get_c_str_literal(c.doc)}", /* tp_doc */
    0,		               /* tp_traverse */
    0,		               /* tp_clear */
    ${c.py_type + "_richcompare"},  /* tp_richcompare */
    0,		               /* tp_weaklistoffset */
    ${c.py_type+"___iter__" if c.iterator else 0},		               /* tp_iter */
    0,		               /* tp_iternext */
    ${c.py_type}_methods,             /* tp_methods */
    0,              /* tp_members */
    ${c.py_type}_getseters,           /* tp_getset */
    0,                         /* tp_base */
    0,                         /* tp_dict */
    0,                         /* tp_descr_get */
    0,                         /* tp_descr_set */
    0,                         /* tp_dictoffset */
    ${"(initproc)%s___init__"%c.py_type if c.constructor else 0},      /* tp_init */
    0,                         /* tp_alloc */
%if c.constructor:
    (newfunc)${c.py_type}_new, /* tp_new */
%else:
    0,                         /* tp_new */
%endif
};

// ----------------------------
// stop class : ${c.py_type}
// ----------------------------

%endfor ## loop on classes

//--------------------- Converters of enums --------------------------

%for en in module.enums :

namespace cpp2py {

template <> struct py_converter<${en.c_name}> {
 static PyObject * c2py(${en.c_name} x) {
   %for n,val in enumerate(en.values[:-1]) :
    if (x == ${val}) return PyUnicode_FromString("${val}");
   %endfor
   return PyUnicode_FromString("${en.values[-1]}"); // last case separate to avoid no return warning of compiler
 }
 static ${en.c_name} py2c(PyObject * ob){
   std::string s = PyUnicode_AsUTF8(ob);
   %for n,val in enumerate(en.values[:-1]) :
    if (s == "${val}") return ${val};
   %endfor
   return ${en.values[-1]};
 }
 static bool is_convertible(PyObject *ob, bool raise_exception) {
   if (!PyUnicode_Check(ob)) {
     if (raise_exception) PyErr_SetString(PyExc_ValueError, "Convertion of C++ enum ${en.c_name} : the object is not a string");
     return false;
   }
   std::string s = PyUnicode_AsUTF8(ob);
   %for n,val in enumerate(en.values) :
    if (s == "${val}") return true;
   %endfor
   if (raise_exception) {
    auto err = "Convertion of C++ enum ${en.c_name} : \nThe string \"" + s +"\" is not in [${','.join([str(x) for x in en.values])}]";
    PyErr_SetString(PyExc_ValueError, err.c_str());
   }
   return false;
 }
};

}
%endfor

//--------------------- define all functions/methods with args, kwds, including constructors -----------------------------

%for py_meth, module_or_class_name, self_c_type in module._all_args_kw_functions() :

 static ${'PyObject*' if not py_meth.is_constructor else 'int'} ${module_or_class_name}_${py_meth.py_name}(PyObject *self, PyObject *args, PyObject *keywds) {

  static constexpr int n_overloads = ${len(py_meth.overloads)};

  [[maybe_unused]] PyObject * py_result; //final result, except for constructor

  std::array<pyref,n_overloads> errors;

  // If no overload, we avoid the err_list and let the error go through (to save some code).
  %for n_overload, overload in enumerate(py_meth.overloads) :
    {// overload ${overload._get_c_signature()}
     %if not overload._dict_call :
     // define the variable to be filled by the parsing method
     // wrapped types are converted to a pointer, other converted types to a value or a view
     %for p, (t,n,d) in enumerate(overload.args) :
     using _type_${p} = std::conditional_t<is_wrapped_v<${t}>, ${t} *, ${t}>;
     _type_${p} __${n} = ${d if d else '_type_%s{}'%p}; // not default for wrapped type please
     %endfor
     static const char *kwlist[] = {${",".join([ '"%s"'%n for t,n,d in overload.args] + ["NULL"])}};
     static const char * format = "${overload._parsing_format()}";
     if (PyArg_ParseTupleAndKeywords(args, keywds, format, const_cast<char **>(kwlist)
       	${"".join([ ('' if t in ['double', 'int'] else ',converter_for_parser<'+t+'>') + ' ,&__%s'%n for t,n,d in overload.args])}))
     %endif
     {
      // redefine the references to remove the * for wrapped type. Allows to use them naturally in C code.
      %if not overload._dict_call :
       %for p, (t,n,d) in enumerate(overload.args) :
       decltype(auto) ${n} = deref_is_wrapped(__${n});
       static_assert(std::is_reference<decltype(${n})>::value || std::is_pointer<decltype(${n})>::value, "internal error");
       %endfor
      %endif

      // FIXME : calling_pattern ---> lambda -> auto , plus de self.
      %if overload.is_method and not overload.is_constructor and not overload.is_static :
      auto & self_c = convert_from_python<${self_c_type}>(self);
      %endif
      %if overload.is_static :
       using self_class = ${self_c_type};
      %endif
      try {
        ${overload._get_calling_pattern1()}
        %if overload.release_GIL_and_enable_signal :
         PyOS_sighandler_t sig = PyOS_getsig(SIGINT);
         cpp2py::signal_handler::start(); // start the C++ signal handler
         Py_BEGIN_ALLOW_THREADS;
	 try {
        %endif
        ${overload._get_calling_pattern()}; // the call is here. It sets up "result" : sets up in the python layer.
        %if overload.release_GIL_and_enable_signal :
	 }
	 catch(...) {
	   // an error has occurred : clean GIL, handler and rethrow
           Py_BLOCK_THREADS; // cf python include, ceval.h, line 72 comments and below.
           cpp2py::signal_handler::stop(); // stop the C++ signal handler
           PyOS_setsig(SIGINT, sig);
           throw; //
	 }
         Py_END_ALLOW_THREADS;
         cpp2py::signal_handler::stop();
         PyOS_setsig(SIGINT, sig);
        %endif
        %if not overload.is_constructor :
 	 %if overload.rtype != "void" :
         py_result = convert_to_python(std::move(result));
         %else:
         Py_INCREF(Py_None);
         py_result = Py_None;
         %endif
        %endif
        return ${'py_result' if not py_meth.is_constructor else '0'};
      }
      %if not py_meth.is_constructor:
      CATCH_AND_RETURN(".. calling C++ overload \n.. ${get_c_str_literal(overload._get_c_signature())} \n.. in implementation of ${'method' if py_meth.is_method else 'function'} ${module_or_class_name}.${py_meth.py_name}", NULL);
      %else:
      CATCH_AND_RETURN (".. in calling C++ overload of constructor :\n.. ${get_c_str_literal(overload._get_c_signature())}",-1);
      %endif
     }
     %if not overload._dict_call :
     else { // the overload does not parse the arguments. Keep the error set by python, for later use, and clear it.
      PyObject * ptype,  *ptraceback, *err; // unused.
      PyErr_Fetch(&ptype, &err, &ptraceback);
      // catch here all non defined errors and print instead generic warning
      if (err == NULL) {
        errors[${n_overload}] = pyref::string("unkown error check converter");
      } else {
        errors[${n_overload}] = pyref{err};
      }
      Py_XDECREF(ptype);
      Py_XDECREF(ptraceback);
     }
     %endif
    } // end overload ${overload._get_c_signature()}
  %endfor # overload

   static const char * overloads_signatures[] = {${'"' + '", "'.join(get_c_str_literal(ov._get_c_signature()) for ov in py_meth.overloads) + '"'}};

   // FIXME Factorize this
   // finally, no overload was successful. Composing a detailed error message, with the reason of failure of each overload
   {
     std::string err_list =
        "Error: no suitable C++ overload found in implementation of ${'method' if py_meth.is_method else 'function'} ${module_or_class_name}.${py_meth.py_name}\n";
     for (int i = 0; i < errors.size(); ++i) {
      err_list = err_list + "\n" + overloads_signatures[i] + " \n failed with the error : \n  ";
      // check if error is not NULL and can be converted at all
      // otherwise unconvertible erros can lead to SegFaults
      if (errors[i] and PyUnicode_Check((PyObject *)errors[i])) {
        err_list += PyUnicode_AsUTF8((PyObject *)errors[i]);
      } else {
        err_list += "Error message was not convertible to raw string.";
      }
      err_list += "\n";
     }
    PyErr_SetString(PyExc_TypeError,err_list.c_str());
   }
   error_return :
   return ${'NULL' if not py_meth.is_constructor else '-1'};
 }
%endfor

// ------------------------------- Loop on all classes ----------------------------------------------------

%for c in module.classes.values() :

//--------------------- define all members  -----------------------------

%for m in c.members:

static PyObject * ${c.py_type}__get_member_${m.py_name} (PyObject *self, void *closure) {
  auto & self_c = convert_from_python<${c.c_type}>(self);
  return convert_to_python(self_c.${m.c_name});
}

%if not m.read_only:
static int ${c.py_type}__set_member_${m.py_name} (PyObject *self, PyObject *value, void *closure) {
  if (value == NULL) { PyErr_SetString(PyExc_TypeError, "Cannot delete the attribute ${m.py_name}"); return -1; }
  if (!convertible_from_python<${m.c_type}>(value, true)) return -1;// exception is set by the converter
  auto & self_c = convert_from_python<${c.c_type}>(self);
  try {
   self_c.${m.c_name} = convert_from_python<${m.c_type}>(value);
  }
  CATCH_AND_RETURN("in setting the attribute '${m.py_name}'",-1);
  return 0;
}
%endif
%endfor ## c.members

//--------------------- define all properties  -----------------------------

%for p in c.properties :

static PyObject * ${c.py_type}__get_prop_${p.name} (PyObject *self, void *closure) {
  %if isinstance(p.getter, str):
    // pure python call
    pyref py_fnt = pyref::module("${p.getter.rsplit('.',1)[0]}").attr("${p.getter.rsplit('.',1)[1]}");
    return py_fnt(self).new_ref();
  %else:
    auto & self_c = convert_from_python<${c.c_type}>(self);
    ${p.getter._get_calling_pattern()}; // defines result, which cannot be void (property would return None ??)
    %if p.getter.rtype != "void" :
       return convert_to_python(std::move(result));
    %else:
       Py_RETURN_NONE;
    %endif
  %endif
}

%if p.setter :
static int ${c.py_type}__set_prop_${p.name} (PyObject *self, PyObject *value, void *closure) {
  if (value == NULL) { PyErr_SetString(PyExc_AttributeError, "Cannot delete the attribute ${p.name}"); return -1; }
  if (!convertible_from_python<${p.setter.args[0][0]}>(value, true)) return -1;
  auto & self_c = convert_from_python<${c.c_type}>(self);
  try {
   %if p.setter._calling_pattern is None:
    self_c.${p.setter.c_name} (convert_from_python<${p.setter.args[0][0]}>(value));
   %else:
    auto value_c = convert_from_python<${p.setter.args[0][0]}>(value);
    ${p.setter._get_calling_pattern()}; // the argument must be called value_c
   %endif
  }
  CATCH_AND_RETURN("in setting the property '${p.name}'",-1);
  return 0;
}
%endif
%endfor ## c.properties


//--------------------- [] implementation  -----------------------------

%if "__len__impl" in c.methods :
static Py_ssize_t ${c.py_type}___len__(PyObject *self) {
 auto & self_c = convert_from_python<${c.c_type}>(self);
 try {
  ${c.methods['__len__impl'].overloads[0]._get_calling_pattern()};
  return result;
 }
 CATCH_AND_RETURN("in calling C++ function for __len__  :\n${get_c_str_literal(overload._get_c_signature())}", -1);
}
%endif

// The __getitem__impl has been implemented with other methods. We redirect the call to the general for (args, kw)
// after rebuilding args.
%if "__getitem__impl" in c.methods :
static PyObject* ${c.py_type}___getitem__(PyObject *self, PyObject *key) {
 pyref args =  (PyTuple_Check(key) ? cpp2py::borrowed(key) : pyref(PyTuple_Pack(1,key)));
 return ${c.py_type}___getitem__impl(self,args,NULL);
}
%endif

%if "__setitem__impl" in c.methods :
static int ${c.py_type}___setitem__(PyObject *self, PyObject *key, PyObject *v) {
 pyref args =  (PyTuple_Check(key) ? cpp2py::borrowed(key) : pyref(PyTuple_Pack(1,key)));
 pyref args2 = PySequence_Concat(args, pyref(PyTuple_Pack(1,v)));
 pyref res =  ${c.py_type}___setitem__impl(self,args2,NULL);
 return (res.is_null() ? -1 : 0); // the function will return None, we want to change to an int
}
%endif

//--------------------- reduce : default case -----------------------------

%if c.serializable is None:

static PyObject* ${c.py_type}___reduce__ (PyObject *self, PyObject *args, PyObject *keywds) {
  PyErr_SetString(PyExc_NotImplementedError, "__reduce__ not implemented");
  return NULL;
}
%endif

//--------------------- reduce h5 -----------------------------

%if c.serializable == "h5" :
 static PyObject* ${c.py_type}___reduce__ (PyObject *self, PyObject *args, PyObject *keywds) {
  auto & self_c = convert_from_python<${c.c_type}>(self);
  pyref r = pyref::module("${module.full_name}").attr("__reduce_reconstructor__${c.py_type}");
  if (r.is_null()) {
   PyErr_SetString(PyExc_ImportError,
                   "Cannot find the reconstruction function ${module.full_name}.__reduce_reconstructor__${c.py_type}");
   return NULL;
  }
  return Py_BuildValue("(NN)", r.new_ref() , Py_BuildValue("(N)", convert_to_python(h5::serialize(self_c))));
 }

 //
 static PyObject* ${c.py_type}___reduce_reconstructor__ (PyObject *self, PyObject *args, PyObject *keywds) {
    PyObject* a1 = PyTuple_GetItem(args,0); // 
    auto a = convert_from_python<std::vector<std::byte>>(a1);
    try {
      return convert_to_python( h5::deserialize<${c.c_type}>(a));
    }
    CATCH_AND_RETURN("in unserialization of object ${c.py_type}",NULL);
  }

%endif

//--------------------- reduce tuple -----------------------------

%if c.serializable == "tuple" :

 // we make the unserialize function using the convertion of a C++ lambda !
 static PyObject* ${c.py_type}___reduce__ (PyObject *self, PyObject *args, PyObject *keywds) {
  auto & self_c = convert_from_python<${c.c_type}>(self);
  pyref r = pyref::module("${module.full_name}").attr("__reduce_reconstructor__${c.py_type}");
  if (r.is_null()) {
   PyErr_SetString(PyExc_ImportError,
                   "Cannot find the reconstruction function ${module.full_name}.__reduce_reconstructor__${c.py_type}");
   return NULL;
  }
  return Py_BuildValue("(NN)", r.new_ref() , reductor{}.apply_to(self_c)); // pyfoo ref ++ by Py_BuildValue
 }

//
 static PyObject* ${c.py_type}___reduce_reconstructor__ (PyObject *self, PyObject *args, PyObject *keywds) {
  try {
    cpp2py::regular_type_if_view_else_type_t<${c.c_type}> result;
    auto r = reconstructor{args};
    result.serialize(r,0);// make sure reconstructor is a friend as boost::serialization::access
    return convert_to_python(std::move(result));
   }
   CATCH_AND_RETURN("in boost unserialization of object ${c.py_type}",NULL);
 }
%endif

//--------------------- reduce repr -----------------------------

%if c.serializable == "repr" :
 static PyObject* ${c.py_type}___reduce__ (PyObject *self, PyObject *args, PyObject *keywds) {
  pyref r = pyref::module("${module.full_name}").attr("__reduce_reconstructor__${c.py_type}");
  if (r.is_null()) {
   PyErr_SetString(PyExc_ImportError, "Cannot find the reconstruction function ${module.full_name}.__reduce_reconstructor__${c.py_type}");
   return NULL;
  }
  return Py_BuildValue("(NN)", r.new_ref(), Py_BuildValue("(N)", PyObject_Repr(self))); // pyfoo ref ++ by Py_BuildValue, the second one is stolen
 }

 //
 static PyObject* ${c.py_type}___reduce_reconstructor__ (PyObject *self, PyObject *args, PyObject *keywds) {
    pyref this_module = PyImport_ImportModule("${module.full_name}");
    if (this_module.is_null()) {
      PyErr_SetString(PyExc_ImportError, "Cannot find the ${module.full_name}");
      return NULL;
    }
    PyObject* global_dict = PyModule_GetDict(this_module); //borrowed
    PyObject* s = PyTuple_GetItem(args,0); //borrowed
    if (!PyUnicode_Check(s)) {
      PyErr_SetString(PyExc_RuntimeError, "Internal error");
      return NULL;
    }
    pyref code       = Py_CompileString(PyUnicode_AsUTF8(s), "nofile", Py_eval_input);
    pyref local_dict = PyDict_New();
    return PyEval_EvalCode(code, global_dict, local_dict);
  }
%endif

//--------------------- repr  -----------------------------

%if c.is_printable :

static PyObject* ${c.py_type}___repr__ (PyObject *self) {
  auto & self_c = convert_from_python<${c.c_type}>(self);
  std::stringstream fs; fs << self_c;
  return PyUnicode_FromString(fs.str().c_str());
}

static PyObject* ${c.py_type}___str__ (PyObject *self) {
  auto & self_c = convert_from_python<${c.c_type}>(self);
  std::stringstream fs; fs << self_c;
  return PyUnicode_FromString(fs.str().c_str());
}

%endif

//--------------------- comparisons  -----------------------------

static PyObject* ${c.py_type}_richcompare (PyObject *a, PyObject *b, int op) {

%if c.comparisons:
 if (!convertible_from_python<${c.c_type}>(a, true)) return NULL;
 if (!convertible_from_python<${c.c_type}>(b, false)) {
   PyErr_SetString(PyExc_TypeError, "Comparison operator : the RHS must be a ${c.c_type}");
   return NULL;
 }
 auto & _a  = convert_from_python<${c.c_type}>(a);
 auto & _b  = convert_from_python<${c.c_type}>(b);
%endif

 switch(op) {

  case Py_EQ:
%if "==" in c.comparisons:
   return convert_to_python(_a ==_b);
%else:
   PyErr_SetString(PyExc_TypeError, "Comparison operator == not implemented for type ${c.py_type}");
   return NULL;
%endif

  case Py_NE:
%if "!=" in c.comparisons:
   return convert_to_python(_a != _b);
%else:
   PyErr_SetString(PyExc_TypeError, "Comparison operator != not implemented for type ${c.py_type}");
   return NULL;
%endif

  case Py_LT:
%if "<" in c.comparisons:
   return convert_to_python(_a < _b);
%else:
   PyErr_SetString(PyExc_TypeError, "Comparison operator < not implemented for type ${c.py_type}");
   return NULL;
%endif

  case Py_GT:
%if ">" in c.comparisons:
   return convert_to_python(_a > _b);
%else:
   PyErr_SetString(PyExc_TypeError, "Comparison operator > not implemented for type ${c.py_type}");
   return NULL;
%endif

  case Py_LE:
%if "<=" in c.comparisons:
   return convert_to_python(_a <= _b);
%else:
   PyErr_SetString(PyExc_TypeError, "Comparison operator <= not implemented for type ${c.py_type}");
   return NULL;
%endif

  case Py_GE:
%if ">=" in c.comparisons:
   return convert_to_python(_a >= _b);
%else:
   PyErr_SetString(PyExc_TypeError, "Comparison operator >= not implemented for type ${c.py_type}");
   return NULL;
%endif

  default:
    break;
 }
 return NULL;
}

//--------------------- hdf5 part -----------------------------
%if not c.hdf5 :

/// hdf5 is not defined for this object, we still but a function + exception for a clear and early error message.
static PyObject* ${c.py_type}___write_hdf5__ (PyObject *self, PyObject *args) {
  PyErr_SetString(PyExc_NotImplementedError, "No hdf5 support for the object ${c.py_type}");
  return NULL;
 }

%else:

// FIXME pass by the ordinary calls....
// FIXME Py_RETURN_NONE : in the usual call.
static PyObject* ${c.py_type}___write_hdf5__ (PyObject *self, PyObject *args) {
  h5::group * gr_ptr;
  const char * key;
  if (!PyArg_ParseTuple(args, "O&s", converter_for_parser<h5::group>, &gr_ptr, &key)) return NULL;
  auto & self_c = convert_from_python<${c.c_type}>(self);
  try {
    h5_write(*gr_ptr, key, self_c);
  }
  CATCH_AND_RETURN("in h5 writing of object ${c.py_type}",NULL);
  Py_RETURN_NONE;
 }

%endif

//--------------------- Arithmetic implementation -----------------------------
  // IMPROVE ERROR TREATEMENT LIKE OTHER ORDINARY METHODS

% for op_name, op in c.number_protocol.items():

%if op.arity == 2:
static PyObject * ${c.py_type}_${op_name} (PyObject* v, PyObject *w){

  %for overload in op.overloads :
  if (convertible_from_python<${overload.args[0][0]}>(v,false) && convertible_from_python<${overload.args[1][0]}>(w,false)) {
   try {
    %if not op_name.startswith("inplace") and not getattr(op, 'treat_as_inplace', False) :
     cpp2py::regular_type_if_view_else_type_t<${overload.rtype}> r =
           convert_from_python<${overload.args[0][0]}>(v) ${overload._get_calling_pattern()} convert_from_python<${overload.args[1][0]}>(w);
     return convert_to_python(std::move(r)); // in two steps to force type for expression templates in C++
    %else:
     convert_from_python<${overload.args[0][0]}>(v) ${overload._get_calling_pattern()} convert_from_python<${overload.args[1][0]}>(w);
     Py_INCREF(v);
     return v;
    %endif
   }
   CATCH_AND_RETURN("in calling C++ overload \n  ${get_c_str_literal(overload._get_c_signature())} \nin implementation of operator ${overload._get_calling_pattern()} ", NULL)
  }
  %endfor
  //PyErr_SetString(PyExc_RuntimeError,"Error: no C++ overload found in implementation of operator ${overload._get_calling_pattern()} ");
  Py_INCREF(Py_NotImplemented);
  return Py_NotImplemented;
}

%elif op.arity == 1:
static PyObject * ${c.py_type}_${op_name} (PyObject *v){

  %for overload in op.overloads :
  if (py_converter<${overload.args[0][0]}>::is_convertible(v,false)) {
   try {
    cpp2py::regular_type_if_view_else_type_t<${overload.rtype}> r =
         ${overload._get_calling_pattern()}(convert_from_python<${overload.args[0][0]}>(v));
    return convert_to_python(std::move(r)); // in two steps to force type for expression templates in C++
   }
   CATCH_AND_RETURN("in calling C++ overload \n  ${get_c_str_literal(overload._get_c_signature())} \nin implementation of operator ${overload._get_calling_pattern()} ", NULL)
  }
  %endfor
  //PyErr_SetString(PyExc_RuntimeError,"Error: no C++ overload found in implementation of operator ${overload._get_calling_pattern()} ");
  Py_INCREF(Py_NotImplemented);
  return Py_NotImplemented;
}

%endif

%endfor

//--------------------- Implementation of iterator by wrapping the C++ -----------------------------

%if c.iterator :

 // the next() method of the iterator
 PyObject* ${c.py_type}__iterator__iternext__(PyObject *self) {
  auto * self_c = (${c.py_type}__iterator *)self;
  if (self_c->iter != self_c->end) {
    %if c.iterator.c_cast_type:
    PyObject *res = convert_to_python((${c.iterator.c_cast_type})(*(self_c->iter)));// make a COPY, I cannot wrap a ref in python !
    %else:
    PyObject *res = convert_to_python(*(self_c->iter));// make a COPY, I cannot wrap a ref in python !
    %endif
    ++self_c->iter;
    return res;
  } else {
    PyErr_SetNone(PyExc_StopIteration);
    return NULL;
  }
 }

// the __iter__ of the main type: return a new iterator. Need to allocate and init it by hand (no init method).
PyObject* ${c.py_type}___iter__(PyObject *self) {
  auto & self_c = convert_from_python<${c.c_type}>(self);
  auto * p = PyObject_New(${c.py_type}__iterator, &${c.py_type}__iteratorType);
  if (!p) return NULL;
  p->container = self;
  Py_INCREF(self); // We keep an owned reference to the container object (e.g. [x for x in g.mesh], if the container is a temporary and the iterator lives longer than its python reference).
  // Initialize the memory of p->iter and p->end with placement new
  new (&p->iter) auto(${c.iterator.begin}(self_c));
  new (&p->end) auto(${c.iterator.end}(self_c));
  return (PyObject *)p;
}

%endif

%endfor  ## Big loop on classes c


// FIXME : remove this
//--------------------- function returning the list of classes, enum wrapped  -----------------------------

 static PyObject* _get_cpp2py_wrapped_class_enums(PyObject *self, PyObject *args, PyObject *keywds) {

  PyObject * d = PyDict_New();

  static const char * cls = "${repr( [ (c.c_type_absolute, c.implement_regular_type_converter) for c in module.classes.values() if c.export])}";
  static const char * ens = "${repr( [ (en.c_name_absolute, en.c_namespace, en.values) for en in module.enums] )}";
  static const char * inclu = "${repr( module.include_list)}";

  PyDict_SetItemString(d, "classes", pyref::string(cls));
  PyDict_SetItemString(d, "enums", pyref::string(ens));
  PyDict_SetItemString(d, "module_name", pyref::string("${module.full_name}"));
  PyDict_SetItemString(d, "includes", pyref::string(inclu));

  return d;
 }

//--------------------- module function table  -----------------------------
// the table of the function of the module...
static PyMethodDef module_methods[] = {
%for pyf in module.functions.values():
    {"${pyf.py_name}", (PyCFunction)${module.name}_${pyf.py_name}, METH_VARARGS| METH_KEYWORDS, "${get_c_str_literal(pyf._generate_doc())}"},
%endfor

%for c in module.classes.values() :
%if c.serializable:
    {"__reduce_reconstructor__${c.py_type}", (PyCFunction)${c.py_type}___reduce_reconstructor__, METH_VARARGS, "Internal " },
%endif
%endfor
    {"_get_cpp2py_wrapped_class_enums", (PyCFunction)_get_cpp2py_wrapped_class_enums, METH_VARARGS, "[Internal] Returns the list of wrapped objects  " },
    {NULL}  /* Sentinel */
};

//--------------------- module struct & init error definition ------------

static struct PyModuleDef ${module.name}_def =
{
    PyModuleDef_HEAD_INIT,
    "${module.name}", /* name of module */
    "${module.doc}", /* module documentation, may be NULL */
    -1,   /* size of per-interpreter state of the module, or -1 if the module keeps state in global variables. */
    module_methods
};

//--------------------- hdf5 helper function -----------------------------
#ifdef H5_INTERFACE_INCLUDED

template <typename T, typename Enable = void>
struct _regular {
  using type = T;
};
template <typename T>
struct _regular<T, std::void_t<typename T::regular_type>> {
  using type = typename T::regular_type;
};

template <typename T> std::function<PyObject *(PyObject *, std::string)> make_py_h5_reader(const char *) {

  auto reader = [](PyObject *h5_gr, std::string const &name) -> PyObject * {
    auto gr = convert_from_python<h5::group>(h5_gr);

    // If T is a view type grab the regular_t of it
    using c_type = typename _regular<T>::type;

    try { // Now call the proper h5_read
      return convert_to_python(T(h5::h5_read<c_type>(gr, name)));
    }
    CATCH_AND_RETURN("in h5 reading of object" + typeid(c_type).name(), NULL);
  };

  return {reader};
}

#endif

//--------------------- module init function -----------------------------
extern "C" __attribute__((visibility("default"))) PyObject* PyInit_${module.name}(void)
{

  if (not cpp2py::check_python_version("${module.name}")) return NULL;

// import numpy iff 'numpy/arrayobject.h' included
#ifdef Py_ARRAYOBJECT_H
    import_array();
#endif

%for c in module.imports :
    PyImport_ImportModule("${c}");
%endfor

    PyObject* m;

%for c in module.classes.values() :
    if (PyType_Ready(&${c.py_type}Type) < 0)
      return NULL;

    %if c.iterator :
    // initializing the ${c.py_type}__iterator
    ${c.py_type}__iteratorType.tp_new = PyType_GenericNew;
    if (PyType_Ready(&${c.py_type}__iteratorType) < 0)
      return NULL;
    Py_INCREF(&${c.py_type}__iteratorType);
    %endif
%endfor

    m = PyModule_Create(&${module.name}_def);
    if (m == NULL)  
      return NULL;


%for c in module.classes.values() :
    Py_INCREF(&${c.py_type}Type);
    PyModule_AddObject(m, "${c.py_type}", (PyObject *)&${c.py_type}Type);
%endfor

#ifdef H5_INTERFACE_INCLUDED
    // hdf5 registration
    pyref module = pyref::module("h5.formats");
    pyref register_class = module.attr("register_class");
%for c in [c for c in module.classes.values() if c.hdf5]:
    {
      pyref h5_reader = convert_to_python(make_py_h5_reader<${c.c_type}>("${c.py_type}"));
      pyref ds = convert_to_python(h5::get_hdf5_format<${c.c_type}>());
      pyref res = PyObject_CallFunctionObjArgs(register_class, (PyObject*)(&${c.py_type}Type), Py_None, (PyObject*)h5_reader, (PyObject*)ds, NULL);
    }
%endfor
#endif

    //std::cout << " INIT Module " << "${module.name}" << std::endl;
    //std::cout << " table ptr count " << cpp2py::conv_table_sptr.use_count() << std::endl;
    // register all the types
    auto &conv_table  = *cpp2py::conv_table_sptr.get();
%for c in module.classes.values() :
    conv_table[std::type_index(typeid(${c.c_type})).name()] = &${c.py_type}Type;
%endfor
    //for(auto & [k, v]: conv_table){
      //std::cout << k << "\n";
    //}

    return m;
}
