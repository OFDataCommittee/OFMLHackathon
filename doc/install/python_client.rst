

First, activate your Python virtual environment:

.. code-block:: bash

    conda activate <env name>

Retrieve and unpack the SmartRedis release tarball:

.. code-block:: bash

    wget <tarball location>
    tar -xf smartredis-0.1.0.tar.gz

Last, build and install the SmartRedis Python client into
your virtual environment.  The pip install process will
automatically download, build, and install SmartRedis
dependencies.

.. code-block:: bash

    pip install -e .

Now, when inside your virtual environment, you should be able to import
the ``Client`` from ``smartredis`` as follows

.. code-block:: python

    Python 3.7.7 (default, May  7 2020, 21:25:33)
    [GCC 7.3.0] :: Anaconda, Inc. on linux
    Type "help", "copyright", "credits" or "license" for more information.
    >>> from smartredis import Client
    >>>
