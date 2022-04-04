#!/bin/bash

OLDPATH=${PATH}
for BNAME in "$@"
do
    export PATH="/opt/python/$BNAME/bin:$OLDPATH"
    pip install .
    py.test
    FILE=tests/failures
    if [ -f "$FILE" ]; then
        cp tests/failures /hostsrc/"$BNAME"_linux_failures.txt
    fi
    
done








