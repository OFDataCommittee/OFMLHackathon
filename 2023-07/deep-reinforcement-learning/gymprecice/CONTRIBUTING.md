# Gym-preCICE Contribution Guidelines

## Development installation

To get the development installation with all the necessary dependencies, please run the following:
```bash
git clone https://github.com/gymprecice/gymprecice.git
cd gymprecice
pip install -e .
pip install -e .[dev]
pre-commit install
```


## Development Process

### Features and Issues

We use GitHub issues to track public bugs. Please ensure your description is
clear and has sufficient instructions to reproduce the issue.

Please avoid using the issue tracker for questions or debugging personal forks. Instead, please use our [Gym-preCICE discussions forum ](https://github.com/gymprecice/gymprecice/discussions)

Please note that if you have any contributions concerning the design and development of current/new active flow control (AFC) `environments`, it should be submitted to our [tutorials](https://github.com/gymprecice/tutorials) repository.

We accept the following types of contributions:
- Bug reports
- Feature requests
- Pull requests for bug fixes
- Documentation improvements
- Performance improvements/issues


### Testing

For pull requests, the project runs a number of tests for the whole project using `pytest`.
- To check all test units locally, run:
```
pytest ./tests
```


### Formatting, Linting, and Type Checking

Gym-preCICE uses [pre-commit](https://pre-commit.com) to run several checks at every new commit. This enforces a common code style across the repository.
We use [black](https://black.readthedocs.io) for code formatting, [flake8](https://flake8.pycqa.org/en/latest/) for linting, [pyright](https://microsoft.github.io/pyright) for type checking, and Pydocstyle to check if all new functions/classes/modules follow the [google docstring style](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html).

Please follow the [development installation instructions](#development-installation) to install the necessary dependencies.

**Note:** You might have to run `pre-commit run --all-files` a few times since formatting tool formats the code initially and fails the first time.


## License

By contributing to Gym-preCICE and its tutorials, you agree that your contributions will be licensed under [the LICENSE file](https://github.com/gymprecice/gymprecice/blob/main/LICENSE) in the root directory of this source tree.
