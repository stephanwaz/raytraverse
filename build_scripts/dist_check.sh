make clean
# make docs
# make coverage

echo -n "proceed to test build (y/n)? "
read -r answer

if [ "$answer" != "${answer#[Yy]}" ] ;then

    python setup.py sdist
    source ~/venv/dev38/bin/activate
    pip wheel . --no-deps -w dist/

    echo -n "proceed to docker build/test (y/n)? "
    read -r answer

    if [ "$answer" != "${answer#[Yy]}" ] ;then

        docker build . --tag rayttest
        docker run -it --name rayt0 --mount type=bind,source="$(pwd)"/dist,target=/wheelhouse rayttest /bin/bash build_scripts/docker_run.sh cp39-cp39
        docker rm rayt0
    fi

    ls -l dist

    echo check files and git commit ...

 fi
