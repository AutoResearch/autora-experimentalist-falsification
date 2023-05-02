# AutoRA Experimentalist Sampler Template

## Quickstart Guide

Install this in an environment using your chosen package manager. In this example we are using virtualenv

Install:
- python (3.8.2 or greater): https://www.python.org/downloads/
- virtualenv: https://virtualenv.pypa.io/en/latest/installation.html

Create a new virtual environment:
```shell
virtualenv venv
```

Activate it:
```shell
source venv/bin/activate
```

Use `pip install` to install the current project (`"."`) in editable mode (`-e`) with dev-dependencies (`[dev]`):
```shell
pip install -e ".[dev]"
```

## Add your contribution 
Your autora-subpackage should include (1) your code implementing the desired **experimentalist**, 
(2) **unit tests** for this experimentalist, and (3) respective **documentation**. 

### Adding the theorist
Add your code to the `src/autora/experimentalist/sampler/your_sampler_name/`

### Adding unit tests
You may also add tests to `tests/test_exp_your_sampler_name.py`

### Adding documentation
You may document your theorist in `docs/index.md`


## Add new dependencies 

In pyproject.toml add the new dependencies under `dependencies`

Install the added dependencies
```shell
pip install -e ".[dev]"
```

## Publishing the package

Update the meta data under `project` in the pyproject.toml file to include name, description, author-name, author-email and version

- Follow the guide here: https://packaging.python.org/en/latest/tutorials/packaging-projects/

Build the package using:
```shell
python -m build
```

Publish the package to PyPI using `twine`:
```shell
twine upload dist/*
```

## Workflows
...