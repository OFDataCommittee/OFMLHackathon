
.. raw:: html

    <div align="center">
        <a href="https://github.com/CrayLabs/SmartSim"><img src="https://github.com/CrayLabs/SmartSim/blob/develop/doc/images/SmartSim_Large.png" width="90%"><img></a>
        <br />
        <br />
    <div display="inline-block">
        <a href="https://github.com/CrayLabs/SmartRedis"><b>Home</b></a>&nbsp;&nbsp;&nbsp;
        <a href="https://www.craylabs.org/build/html/installation.html"><b>Install</b></a>&nbsp;&nbsp;&nbsp;
        <a href="https://www.craylabs.org/build/html/overview.html"><b>Documentation</b></a>&nbsp;&nbsp;&nbsp;
        <a href="https://github.com/CrayLabs/SmartRedis/releases/download/v0.1.0/smartredis-0.1.0.tar.gz"><b>Download</b></a>&nbsp;&nbsp;&nbsp;
        <a href="https://craylabs.slack.com/ssb/redirect"><b>Slack</b></a>&nbsp;&nbsp;&nbsp;
        <a href="https://github.com/CrayLabs"><b>Cray Labs</b></a>&nbsp;&nbsp;&nbsp;
      </div>
        <br />
        <br />
    </div>

.. |license| image:: https://img.shields.io/github/license/CrayLabs/SmartRedis
    :target: https://github.com/CrayLabs/SmartRedis/blob/master/LICENSE.md
    :alt: License
    
.. |language| image:: https://img.shields.io/github/languages/top/CrayLabs/SmartRedis
    :alt: Language

.. |tag| image:: https://img.shields.io/github/v/tag/CrayLabs/SmartRedis  
    :alt: GitHub tag (latest by date)

| |License|  |Language|  |tag| 

----------


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
