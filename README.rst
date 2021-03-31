==========
SmartRedis
==========

SmartRedis is a collection of Redis clients that support
RedisAI capabilities and include additional
features for high performance computing (HPC) applications.
SmartRedis provides clients in the following languages:

.. list-table::
   :widths: 20 20
   :header-rows: 1

   * - Language
     - Version/Standard
   * - Python
     - 3.7+
   * - C++
     - C++17
   * - C
     - C99
   * - Fortran
     - Fortran 2018

Using SmartRedis
================

To install and run SmartRedis, follow the instructions in
the documentation. To build the documentation, install
the python dependencies (including sphinx) with the
command:

.. code-block:: bash

    pip install -r requirements-dev.txt

In the top level of the SmartRedis directory, execute the
following command to build the documentation:

.. code-block:: bash

  make docs

Open ``doc/_build/html/index.html`` in a browser to view
the documentation.
