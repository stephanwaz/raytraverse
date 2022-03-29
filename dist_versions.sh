make clean
source ~/venv/dev39/bin/activate
python setup.py sdist bdist_wheel --plat-name macosx-10.9-x86_64
source ~/venv/dev38/bin/activate
python setup.py bdist_wheel --plat-name macosx-10.9-x86_64
source ~/venv/dev/bin/activate
python setup.py bdist_wheel --plat-name macosx-10.9-x86_64
ls -l dist
