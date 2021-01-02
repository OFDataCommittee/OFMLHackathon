====
SILC
====

The SmartSim Infrastructure Library Clients (SILC) are a library
of redis-based clients built primarily for use with SmartSim.

.. list-table::
   :widths: 15 10 30
   :header-rows: 1

   * - Language
     - Version/Standard
     - Status
   * - Python
     - 3.7+
     - In development
   * - C++
     - C++17
     - Stable
   * - C
     - C99
     - Stable
   * - Fortran
     - Fortran 2003 +
     - In development

Project Status
==============

Currently the C++, and C are the only full implementations of the SILC clients.
The Fortran and Python partially cover the API and will soon be feature-complete.


Using SILC
==========

Installation
------------

To install SILC, follow the instructions in the documentation. To build
the documentation, install the python dependencies (including sphinx) and
in the top level of the SILC directory execute the following

.. code-block:: bash

  make docs

Then open ``doc/_build/html/index.html`` in a browser

Examples
--------

For examples of how to build and use the c++ client, refer to the c++ test
directory in this repository.

