

First, activate your Python virtual environment.

.. code-block:: bash

    conda activate <env name>

Retrieve, and unpack the SmartRedis release tarball

.. code-block:: bash

    wget <tarball location>
    tar -xf smartredis-0.1.0

Next, build the dependencies of SmartRedis. For more information
on the dependencies of SmartRedis, see the ``From Source`` instructions.

.. code-block:: bash

    cd smartredis-0.1.0
    make deps
    source setup_env.sh

Last, build and install the SmartRedis Python client into your virtual environment.
Make sure you didn't change terminal sessions between the last step and this step.

.. code-block:: bash

    make pyclient
    pip install -e .

Now, when inside your virtual environment, you should be able to import
the ``Client`` from ``smartredis`` as follows

.. code-block:: python

    Python 3.7.7 (default, May  7 2020, 21:25:33)
    [GCC 7.3.0] :: Anaconda, Inc. on linux
    Type "help", "copyright", "credits" or "license" for more information.
    >>> from smartredis import Client
    >>>
