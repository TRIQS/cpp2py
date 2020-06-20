#pragma once
#include <Python.h>
#include <numpy/arrayobject.h>

#include <vector>
#include <complex>

#include "pyref.hpp"

namespace cpp2py {

  using v_t = std::vector<long>;

  // the basic information for a numpy array
  struct numpy_proxy {
    int rank          = 0;
    long element_type = 0;
    void *data        = nullptr;
    bool is_const     = false;
    v_t extents, strides;
    PyObject *base = nullptr; // The ref. counting guard typically

    // Returns a new ref (or NULL if failure) with a new numpy.
    // If failure, return null with the Python exception set
    PyObject *to_python();
  };

  // From a numpy, extract the info. Better than a constructor, I want to use the aggregate constructor of the struct also.
  numpy_proxy make_numpy_proxy(PyObject *);

  // Make a copy in Python with the given rank and element_type
  // If failure, return null with the Python exception set
  PyObject *make_numpy_copy(PyObject *obj, int rank, long elements_type);

  //
  template <typename T> inline constexpr long npy_type = -1;
  template <typename T> inline constexpr bool has_npy_type = (npy_type<T> >= 0);

#define NPY_CONVERT(C, P) template <> inline constexpr long npy_type<C> = P;
  NPY_CONVERT(pyref, NPY_OBJECT)
  NPY_CONVERT(bool, NPY_BOOL)
  NPY_CONVERT(char, NPY_STRING)
  NPY_CONVERT(signed char, NPY_BYTE)
  NPY_CONVERT(unsigned char, NPY_UBYTE)
  NPY_CONVERT(short, NPY_SHORT)
  NPY_CONVERT(unsigned short, NPY_USHORT)
  NPY_CONVERT(int, NPY_INT)
  NPY_CONVERT(unsigned int, NPY_UINT)
  NPY_CONVERT(long, NPY_LONG)
  NPY_CONVERT(unsigned long, NPY_ULONG)
  NPY_CONVERT(long long, NPY_LONGLONG)
  NPY_CONVERT(unsigned long long, NPY_ULONGLONG)
  NPY_CONVERT(float, NPY_FLOAT)
  NPY_CONVERT(double, NPY_DOUBLE)
  NPY_CONVERT(long double, NPY_LONGDOUBLE)
  NPY_CONVERT(std::complex<float>, NPY_CFLOAT)
  NPY_CONVERT(std::complex<double>, NPY_CDOUBLE)
  NPY_CONVERT(std::complex<long double>, NPY_CLONGDOUBLE)
#undef NPY_CONVERT

} // namespace cpp2py
