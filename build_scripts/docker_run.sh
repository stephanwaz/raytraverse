#!/bin/bash
set -e -u -x

function repair_wheel {
    wheel="$1"
    if ! auditwheel show "$wheel"; then
        echo "Skipping non-platform wheel $wheel"
    else
        auditwheel repair "$wheel" --plat "$AUDITWHEEL_PLAT" -w /wheelhouse/
    fi
}



OLDPATH=${PATH}
for BNAME in "$@"
do
    export PATH="/opt/python/$BNAME/bin:$OLDPATH"
    pip wheel . --no-deps -w dist/
    pip install dist/raytraverse-*-"$BNAME"-*.whl
    pytest tests/test_cr*.py
    FILE=tests/failures
    if [ -f "$FILE" ]; then
        cp tests/failures /wheelhouse/"$BNAME"_linux_failures.txt
    fi
    repair_wheel dist/raytraverse-*-"$BNAME"-*.whl
    
done








