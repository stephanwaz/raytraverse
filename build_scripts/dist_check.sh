make clean
make docs
make coverage

echo -n "proceed to test build (y/n)? "
read -r answer

if [ "$answer" != "${answer#[Yy]}" ] ;then

    python setup.py sdist
    source ~/venv/dev38/bin/activate
    python setup.py bdist_wheel

    echo -n "proceed to docker build/test (y/n)? "
    read -r answer

    if [ "$answer" != "${answer#[Yy]}" ] ;then

        docker image tag raytdev:3.9 raytdev:current
        docker build . --tag rayttest
        docker run --name rayt0 --mount type=bind,source="$(pwd)"/dist,target=/wheelhouse rayttest
        docker stop -t 0 rayt0
        docker rm rayt0
        docker image rm rayttest
    fi

    ls -l dist

    echo check files and git commit ...

 fi
