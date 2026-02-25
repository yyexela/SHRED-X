# SHRED-X

**A modular PyTorch framework for Shallow Recurrent Decoders** -- reconstruct full spatiotemporal fields from sparse sensor measurements.

Website: [https://yyexela.github.io/SHRED-X/](https://yyexela.github.io/SHRED-X/)

## Documentation

Full documentation is hosted at **[yyexela.github.io/SHRED-X](https://yyexela.github.io/SHRED-X/)** and includes:

- [Getting Started](https://yyexela.github.io/SHRED-X/getting_started.html) -- installation and quickstart
- [Tutorials](https://yyexela.github.io/SHRED-X/tutorials.html) -- end-to-end notebooks
- [API Reference](https://yyexela.github.io/SHRED-X/api.html) -- complete module/class/function docs
- [Contributing](https://yyexela.github.io/SHRED-X/contributing.html) -- development setup and workflow


## Installation

```bash
git clone https://github.com/yyexela/SHRED-X.git
cd SHRED-X

pyenv install 3.13.7
pyenv local 3.13.7
python -m venv ~/.virtualenvs/shredx
source ~/.virtualenvs/shredx/bin/activate
pip install -e .
```

For development (tests, linting, type checking, docs):

```bash
pip install -e ".[dev]"
```

## Citations

If you use SHRED in your research, please cite the relevant papers listed on the [Citations](https://yyexela.github.io/SHRED-X/citations.html) page.

## Contributing

See the [Contributing guide](https://yyexela.github.io/SHRED-X/contributing.html) for details.
