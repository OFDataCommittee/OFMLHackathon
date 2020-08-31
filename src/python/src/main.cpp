#include "pybind11/pybind11.h"

#include "xtensor/xmath.hpp"
#include "xtensor/xarray.hpp"

#define FORCE_IMPORT_ARRAY
#include "xtensor-python/pyarray.hpp"
#include "xtensor-python/pyvectorize.hpp"

#include <iostream>
#include <numeric>
#include <string>
#include <sstream>


// our headers
#include "silc/silc.hpp"
#include "silc/silc_config.hpp"

namespace py = pybind11;



namespace silc {


    // implementation in def_myclass.cpp
    void def_class(py::module & m);

    // implementation in def_myclass.cpp
    void def_build_config(py::module & m);

    // implementation in def.cpp
    void def_build_config(py::module & m);

}


// Python Module and Docstrings
PYBIND11_MODULE(_silc , module)
{
    xt::import_numpy();

    module.doc() = R"pbdoc(
        _silc  python bindings

        .. currentmodule:: _silc 

        .. autosummary::
           :toctree: _generate

           BuildConfiguration
           MyClass
    )pbdoc";

    silc::def_build_config(module);
    silc::def_class(module);

    // make version string
    std::stringstream ss;
    ss<<SILC_VERSION_MAJOR<<"."
      <<SILC_VERSION_MINOR<<"."
      <<SILC_VERSION_PATCH;
    module.attr("__version__") = ss.str().c_str();
}