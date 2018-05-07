#include "./misc.hpp"
namespace cpp2py {

  template <typename T> std::function<PyObject *(PyObject *, std::string)> make_py_h5_reader(const char *) {

    auto reader = [](PyObject *h5_gr, std::string const &name) -> PyObject * {
      auto gr = convert_from_python<triqs::h5::group>(h5_gr);
      // declare the target C++ object, with special case if it is a view...
      using c_type = triqs::regular_type_if_view_else_type_t<T>;
      try {                                                                // now read
        return convert_to_python(T(triqs::h5::h5_read<c_type>(gr, name))); // cover the view and value case
      }
      CATCH_AND_RETURN("in h5 reading of object ${c.py_type}", NULL);
      return NULL; // unused
    };

    return {reader};
  }
} // namespace cpp2py
