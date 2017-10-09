// DO NOT EDIT
// --- C++ Python converter for ${c.spelling}
#include <cpp2py/converters/vector.hpp>
#include <cpp2py/converters/string.hpp>
#include <algorithm>

namespace cpp2py { 

template <> struct py_converter<${c.spelling}> {
 static PyObject *c2py(${c.spelling} const & x) {
  PyObject * d = PyDict_New(); 
  %for m in c_members :
  PyDict_SetItemString( d, ${name_format('"%s"'%m.spelling)}, convert_to_python(x.${m.spelling}));
  %endfor
  return d;
 }

 template <typename T, typename U> static void _get_optional(PyObject *dic, const char *name, T &r, U const &init_default) {
  if (PyDict_Contains(dic, pyref::string(name)))
   r = convert_from_python<T>(PyDict_GetItemString(dic, name));
  else
   r = init_default;
 }

 template <typename T> static void _get_optional(PyObject *dic, const char *name, T &r) {
  if (PyDict_Contains(dic, pyref::string(name)))
   r = convert_from_python<T>(PyDict_GetItemString(dic, name));
  else
   r = T{};
 }

 static ${c.spelling} py2c(PyObject *dic) {
  ${c.spelling} res;
  %for m, m_initializer in [(m,CL.get_member_initializer(m)) for m in  c_members]:
  %if m_initializer == '' : 
  res.${m.spelling} = convert_from_python<${m.type.spelling}>(PyDict_GetItemString(dic, "${m.spelling}"));
  %else:
  _get_optional(dic, ${name_format_q(m.spelling)}, res.${name_format(m.spelling)} ${',' + m_initializer if m_initializer !="{}" else ''});
  %endif
  %endfor
  return res;
 }

 template <typename T>
 static void _check(PyObject *dic, std::stringstream &fs, int &err, const char *name, const char *tname) {
  if (!convertible_from_python<T>(PyDict_GetItemString(dic, name), false))
   fs << "\n" << ++err << " The parameter " << name << " does not have the right type : expecting " << tname
      << " in C++, but got '" << PyDict_GetItemString(dic, name)->ob_type->tp_name << "' in Python.";
 }

 template <typename T>
 static void _check_mandatory(PyObject *dic, std::stringstream &fs, int &err, const char *name, const char *tname) {
  if (!PyDict_Contains(dic, pyref::string(name)))
   fs << "\n" << ++err << " Mandatory parameter " << name << " is missing.";
  else _check<T>(dic,fs,err,name,tname);
 }

 template <typename T>
 static void _check_optional(PyObject *dic, std::stringstream &fs, int &err, const char *name, const char *tname) {
  if (PyDict_Contains(dic, pyref::string(name))) _check<T>(dic, fs, err, name, tname);
 }

 static bool is_convertible(PyObject *dic, bool raise_exception) {
  if (dic == nullptr or !PyDict_Check(dic)) {
   if (raise_exception) { PyErr_SetString(PyExc_TypeError, "The function must be called with named arguments");}
   return false;
  }  
  std::stringstream fs, fs2; int err=0;

#ifndef TRIQS_ALLOW_UNUSED_PARAMETERS
  std::vector<std::string> ks, all_keys = {${','.join('"%s"'%m.spelling for m in c_members)}};
  pyref keys = PyDict_Keys(dic);
  if (!convertible_from_python<std::vector<std::string>>(keys, true)) {
   fs << "\nThe dict keys are not strings";
   goto _error;
  }
  ks = convert_from_python<std::vector<std::string>>(keys);
  for (auto & k : ks)
   if (std::find(all_keys.begin(), all_keys.end(), k) == all_keys.end())
    fs << "\n"<< ++err << " The parameter '" << k << "' is not recognized.";
#endif

  %for m in c_members :
  %if CL.get_member_initializer(m) == '' : 
  _check_mandatory<${type_format(m.type.spelling)}>(dic, fs, err, ${name_format_q(m.spelling)}, "${m.type.spelling}"); 
  %else:
  _check_optional <${type_format(m.type.spelling)}>(dic, fs, err, ${name_format_q(m.spelling)}, "${m.type.spelling}");
  %endif
  %endfor
  if (err) goto _error;
  return true;
  
 _error: 
   fs2 << "\n---- There " << (err > 1 ? "are " : "is ") << err<< " error"<<(err >1 ?"s" : "")<< " in Python -> C++ transcription for the class ${c.spelling}\n" <<fs.str();
   if (raise_exception) PyErr_SetString(PyExc_TypeError, fs2.str().c_str());
  return false;
 }
};

}
