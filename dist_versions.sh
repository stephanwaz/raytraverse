make clean
python setup.py sdist
# source ~/venv/dev39/bin/activate
# python setup.py bdist_wheel --plat-name macosx-10.9-x86_64
# source ~/venv/dev38/bin/activate
# python setup.py bdist_wheel --plat-name macosx-10.9-x86_64
# source ~/venv/dev/bin/activate
# python setup.py bdist_wheel --plat-name macosx-10.9-x86_64



# docker image tag raytdev:3.10 raytdev:current
# docker build . --tag rayttest
# docker run --name rayt0 --mount type=bind,source="$(pwd)"/dist,target=/wheelhouse rayttest
# docker stop -t 0 rayt0
# docker rm rayt0
# docker image rm rayttest


for i in 7 8 9 10
do
    docker image tag raytdev:3."$i" raytdev:current
    docker build . --tag rayttest
    docker run --name rayt0 --mount type=bind,source="$(pwd)"/dist,target=/wheelhouse rayttest
    docker stop -t 0 rayt0
    docker rm rayt0
    docker image rm rayttest
done

# ls -l dist