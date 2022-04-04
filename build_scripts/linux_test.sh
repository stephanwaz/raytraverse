python setup.py sdist
docker build . --tag rayttest
docker run -it --name rayt0 --mount type=bind,source="$(pwd)",target=/hostsrc rayttest /bin/bash build_scripts/docker_test.sh cp39-cp39
docker rm rayt0