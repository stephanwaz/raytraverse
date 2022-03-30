BNAME=$(basename dist/*.whl .whl)
py.test 1> test_log.txt 
cp test_log.txt ../../wheelhouse/"$BNAME"_test_log.txt
cp tests/failures ../../wheelhouse/"$BNAME"_failures.txt
cp dist/*.whl ../../wheelhouse/
