Contributing to SHRED-X
=======================

Environment
------------

The environment can be created by running::

   pyenv install 3.13.7
   pyenv local 3.13.7
   python -m venv ~/.virtualenvs/shredx
   source ~/.virtualenvs/shredx/bin/activate
   pip install -e .[dev]

Nox Taskrunner
--------------

We use `Nox <https://nox.thea.codes/en/stable/index.html>`__ to automate all tasks in our package. Specifically, running ``nox`` runs all tests, lints and formats the code, performs typechecking, and builds the documentation. It is also used for the continuous integration pipeline to do all the aforementioned in Github Actions.

To run a specific Nox task, run ``nox --list`` to see available tasks and then ``nox --no-venv --no-install -s <task>`` to run the task.

Pre-commit
----------

We use `pre-commit <https://pre-commit.com/>`__ to verify all code is linted, formatted, and type-checked before committed. This ensures a smoother code development process. Use ``pre-commit install`` to install the appropriate hooks and verify it runs with ``pre-commit run --all-files``.

Linting and Formatting
----------------------

Linting ensures that the code logic makes sense and nothing is being called in a way that would surely break. For this, we use `Ruff <https://docs.astral.sh/ruff/>`__. We also use Ruff for code formatting, ensuring a standard format across all code. There is a Ruff VSCode extension that can be used to help with linting and formatting.

Type checking
-------------

In this repository, we use `Pyrefly <https://pyrefly.org/>`__ to check all functions are called with their specified types. We also use `jaxtyping <https://docs.kidger.site/jaxtyping/api/array/>`__ to allow for clear function typing for PyTorch array dimensions. There is a Pyrefly VSCode extension that can be used to help with type checking.

Building/Serving Documentation
------------------------------

We use `Sphinx <https://www.sphinx-doc.org/en/master/index.html>`__ to build our documentation. Make sure that the ``sphinx`` package is installed in your python environment. The source files are located in ``doc/``. To build the documentation, run ``nox --no-venv --no-install -s build_docs``.

To preview documentation files, open ``docs/build/html/index.html`` locally in your browser. If you're developing on a server, first run ``python -m http.server 8000`` in ``docs/build/html`` and then on your local computer run ``ssh -L 8080:localhost:8000 <server_address>``. This will turn the server into an HTTP server and opening ``localhost:8080`` in a browser will show the generated documentation.

Adding to Documentation
-----------------------

We use `Sphinx <https://www.sphinx-doc.org/en/master/index.html>`__ as our documentation generator. To add to the documentation, add to the ``source/`` directory. All files in this directory will be included in the documentation. Please read the `Sphinx documentation <https://www.sphinx-doc.org/en/master/usage/quickstart.html>`__ for more information on how to write documentation.
