.PHONY: clean clean-test clean-pyc clean-build docs help
.DEFAULT_GOAL := help

define BROWSER_PYSCRIPT
import os, webbrowser, sys

try:
	from urllib import pathname2url
except:
	from urllib.request import pathname2url

webbrowser.open("file://" + pathname2url(os.path.abspath(sys.argv[1])))
endef
export BROWSER_PYSCRIPT

define PRINT_HELP_PYSCRIPT
import re, sys

for line in sys.stdin:
	match = re.match(r'^([a-zA-Z_-]+):.*?## (.*)$$', line)
	if match:
		target, help = match.groups()
		print("%-20s %s" % (target, help))
endef
export PRINT_HELP_PYSCRIPT

BROWSER := python -c "$$BROWSER_PYSCRIPT"

help:
	@python -c "$$PRINT_HELP_PYSCRIPT" < $(MAKEFILE_LIST)

clean: clean-build clean-pyc clean-test ## remove all build, test, coverage and Python artifacts

clean-build: ## remove build artifacts
	rm -fr build/
	rm -fr var/
	rm -fr dist/
	rm -fr .eggs/
	rm -rf _skbuild/
	# find . -name '*.egg-info' -exec rm -fr {} +
	find . -name '*.egg' -exec rm -f {} +
	find . -name '*.o' -exec rm -f {} +

clean-pyc: ## remove Python file artifacts
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f {} +
	find . -name '__pycache__' -exec rm -fr {} +

clean-test: ## remove test and coverage artifacts
	rm -fr .tox/
	rm -f .coverage*
	rm -fr htmlcov/
	rm -fr .pytest_cache

lint: ## check style with flake8
	flake8 raytraverse tests

test: ## run tests quickly with the default Python
	py.test

slowtest: ## run tests including slow decorator
	py.test --slow

coverage: ## check code coverage quickly with the default Python
	pytest --cov=raytraverse tests
	#pytest --cov=raytraverse --cov-append --slowtest tests/test_skycalc.py tests/test_cli.py
	coverage html
	$(BROWSER) htmlcov/index.html

coverall: coverage
	coveralls

docs: cdocs pdocs showdocs## generate Sphinx HTML documentation, including API docs


cdocs: ## generate Sphinx HTML documentation, including API docs
	$(MAKE) -C docs/cdocs clean
	$(MAKE) -C docs/cdocs html
	python docs/cdocs/static_html.py
	$(MAKE) -C docs/cdocs clean

pdocs:
	# rm -f docs/raytraverse.rst
	# rm -f docs/modules.rst
	# sphinx-apidoc -o docs/ raytraverse
	$(MAKE) -C docs clean
	$(MAKE) -C docs html

showdocs:
	$(BROWSER) docs/_build/html/index.html

servedocs: docs ## compile the docs watching for changes
	watchmedo shell-command -p '*.rst' -c '$(MAKE) -C docs html' -R -D .

#release: dist ## package and upload a release
#	twine upload dist/*

dist: clean ## builds source and wheel package
	python setup.py sdist bdist_wheel
	ls -l dist

install: clean ## install the package to the active Python's site-packages
	python setup.py install

uml:
	pyreverse raytraverse -o pdf -f OTHER

log:
	git log --graph --all --date=short --pretty=tformat:"%w(80,0,20)%C(auto)%h %C(red bold)%ad:%C(auto)%d%n%w(80,8,8)%s"
