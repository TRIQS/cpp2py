Cpp2Py is the Python-C++ interfacing tool of the [TRIQS](https://triqs.github.io) project, provided here as a standalone project.

Installation
============

To install Cpp2Py, follow the installation steps:

```bash
git clone  https://github.com/TRIQS/cpp2py.git cpp2py
mkdir build && cd build
cmake -DCMAKE_INSTALL_PREFIX=INSTALL_DIR ../cpp2py
make && make install
```

This installs the library in `INSTALL_DIR`.
In order to make Cpp2Py available in your current environment you should run

```bash
source INSTALL_DIR/share/cpp2pyvars.sh
```


Example
=======

Make sure that you have loaded Cpp2Py into your environment as instructed above.
Created a C++ source file `mymodule.hpp` in a folder `SRC`:

```c++
///A wonderful little class
class myclass{
  int a, b;

  public:
    myclass(int a_, int b_) : a(a_), b(b_) {}

    ///getter for member a
    int get_a() const { return a;}
};
```

In the same folder, create a file `CMakeLists.txt`:

```cmake
cmake_minimum_required(VERSION 3.0.2)
find_package(Cpp2Py REQUIRED)

add_cpp2py_module(mymodule)
target_compile_options(mymodule PRIVATE -std=c++17)
target_include_directories(mymodule PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
```

Then, in the `SRC` folder, issue the command

```
c++2py mymodule.hpp
```

This creates a file `mymodule_desc.py`.

Exit the `SRC` folder and create a `BUILD` folder. Then, issue the following commands:

```bash
cd BUILD
cmake ../SRC
make
```

In the `BUILD` dir, you should see a `mymodule.so` file. You can now use your c++ class in Python:

```python
import mymodule
A = mymodule.Myclass(4,5)
print(A.get_a())
```

By convention, c++ classes of the type `my_little_class` are converted in python classes of the type `MyLittleClass`.

License
===============

Before you proceed, make sure you have read the `LICENSE.txt` file.

Enjoy!

The TRIQS team
