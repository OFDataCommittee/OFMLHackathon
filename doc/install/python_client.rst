

First, activate your Python virtual environment:

.. code-block:: bash

    conda activate <env name>

Install SmartRedis Python client from PyPI:

.. code-block:: bash

    pip install smartredis

Developers installing the Python client from PyPI
can install the SmartRedis client with additional
dependencies for testing and documentation with:

.. code-block:: bash

    pip install smartredis[dev]
    # or if using ZSH
    pip install smartredis\[dev\]

Now, when inside your virtual environment, you should be able to import
the ``Client`` from ``smartredis`` as follows

.. code-block:: python

    Python 3.7.7 (default, May  7 2020, 21:25:33)
    [GCC 7.3.0] :: Anaconda, Inc. on linux
    Type "help", "copyright", "credits" or "license" for more information.
    >>> from smartredis import Client
    >>>
