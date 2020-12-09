
MAKEFLAGS += --no-print-directory

# Do not remove this block. It is used by the 'help' rule when
# constructing the help output.
# help:
# help: SILC Makefile help
# help:

# help: help                           - display this makefile's help information
.PHONY: help
help:
	@grep "^# help\:" Makefile | grep -v grep | sed 's/\# help\: //' | sed 's/\# help\://'


# help: clean                          - clean all files using .gitignore rules
.PHONY: clean
clean:
	@git clean -X -f -d


# help: scrub                          - clean all files, even untracked files
.PHONY: scrub
scrub:
	git clean -x -f -d


# help: test                           - run tests
.PHONY: test
test:
	@python -m pytest ./tests


# help: test-v                         - run tests [verbosely]
.PHONY: test-v
test-v:
	@python -m pytest -vv ./tests


# help: test-cov                       - perform test coverage checks
.PHONY: test-cov
test-cov:
	@python -m pytest --cov=./src/python/module/silc/ -vv ./tests

# help: format                         - perform code style format
.PHONY: format
format:
	@black ./src/python/module/silc ./tests/python/


# help: check-format                   - check code format compliance
.PHONY: check-format
check-format:
	@black --check ./src/python/module/silc ./tests/python/


# help: sort-imports                   - apply import sort ordering
.PHONY: sort-imports
sort-imports:
	@isort ./src/python/module/silc ./tests/python/ --profile black


# help: check-sort-imports             - check imports are sorted
.PHONY: check-sort-imports
check-sort-imports:
	@isort  ./src/python/module/silc ./tests/python/ --check-only --profile black


# help: style                          - perform code style format
.PHONY: style
style: sort-imports format


# help: check-style                    - check code style compliance
.PHONY: check-style
check-style: check-sort-imports check-format


# help: check-lint                     - run static analysis checks
.PHONY: check-lint
check-lint:
	@pylint --rcfile=.pylintrc ./src/python/module/silc ./tests/python


# help: docs                           - generate project documentation
.PHONY: docs
docs: coverage
	@cd docs; make html


