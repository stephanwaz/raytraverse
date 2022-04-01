# build linux images with different python versions and requirements.
# tag with current to use in buid process

# docker build -f Dockerfile310 --tag raytdev:3.10 .
# docker build -f Dockerfile37 --tag raytdev:3.7 .
docker build -f Dockerfile38 --tag raytdev:3.8 .
docker build -f Dockerfile39 --tag raytdev:3.9 .