Contributing
============

Thank you for your interest in contributing to SHRED-X!  This guide
covers the development environment, tooling, and workflow.


Development Setup
-----------------

.. code-block:: bash

   git clone https://github.com/yyexela/SHRED-X.git
   cd SHRED-X

   pyenv install 3.13.7
   pyenv local 3.13.7
   python -m venv ~/.virtualenvs/shredx
   source ~/.virtualenvs/shredx/bin/activate
   pip install -e ".[dev]"

Install the pre-commit hooks so that every commit is automatically
linted, formatted, and type-checked:

.. code-block:: bash

   pre-commit install
   pre-commit run --all-files   # verify everything passes


Nox Task Runner
---------------

`Nox <https://nox.thea.codes/en/stable/index.html>`__ orchestrates all
CI tasks.  Run ``nox --list`` to see available sessions, or invoke a
specific one. View ``noxfile.py`` for more details.

.. code-block:: bash

   nox --no-venv --no-install -s <session>


Linting & Formatting
---------------------

We use `Ruff <https://docs.astral.sh/ruff/>`__ for both linting and
formatting.  The `Ruff VS Code extension
<https://marketplace.visualstudio.com/items?itemName=charliermarsh.ruff>`__
provides real-time feedback in the editor.


Type Checking
-------------

`Pyrefly <https://pyrefly.org/>`__ verifies that all function calls
respect their declared types.  Tensor dimension annotations use
`jaxtyping <https://docs.kidger.site/jaxtyping/api/array/>`__ for
readable shape documentation.


Building Documentation
----------------------

We use Sphinx to build the documentation. Learn more at `sphinx-doc.org <https://www.sphinx-doc.org/en/master/index.html>`__.

.. code-block:: bash

   nox --no-venv --no-install -s build_docs

The generated site lands in ``docs/build/html/``.  Open
``docs/build/html/index.html`` in a browser to preview.

If you are developing on a remote server:

.. code-block:: bash

   # On the server
   python -m http.server 8000 -d docs/build/html

   # On your local machine
   ssh -L 8080:localhost:8000 <server>

Then open ``http://localhost:8080`` in your browser.


Adding Documentation
--------------------

Source files live in ``docs/source/``.  Add or edit ``.rst`` files there
and ensure they are referenced from a ``toctree``.  API docs are
auto-generated from docstrings via ``autosummary`` -- just write good
docstrings and they will appear automatically.

Please follow `numpydoc
<https://numpydoc.readthedocs.io/en/latest/format.html>`__ style for all
docstrings.
