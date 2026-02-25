Getting Started
===============

Installation
------------

**Prerequisites** -- Python 3.13+ and a working ``pip``. GPU support
requires a CUDA-enabled PyTorch installation
(see `pytorch.org <https://pytorch.org/get-started/locally/>`_).

Install from source (recommended while the package is in development):

.. code-block:: bash

   git clone https://github.com/yyexela/SHRED-X.git
   cd SHRED-X

   pyenv install 3.13.7
   pyenv local 3.13.7
   python -m venv ~/.virtualenvs/shredx
   source ~/.virtualenvs/shredx/bin/activate
   pip install -e .

For development (tests, linting, docs):

.. code-block:: bash

   pip install -e ".[dev]"

Next Steps
----------

* :doc:`tutorials` -- end-to-end notebook with Sea-Surface Temperature data.
* :doc:`api` -- full API reference.
