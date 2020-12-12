============
Installation
============

Currently you can easily installed with setuptools and pip.

First make sure that the current versions of setuptools and wheels are isntalled. Then, execute the setupfile via:

.. code-block:: bash

    python3 setup.py bdist_wheel

Afterwards switch to the newly created dist folder and install textclust with pip

.. code-block:: bash

    cd dist
    pip3 install textClustPy-VERSION-py3-none-any.whl

Alternatively you can simply call install.sh

.. code-block:: bash

    sh install.sh

Thats it! You can now use textClust.