# syntax=docker/dockerfile:1

# docker build . --tag raytraverse:latest --tag raytraverse:v1.3.0
# docker run -it --name rayt --mount type=bind,source="$(pwd)",target=/working raytraverse /bin/bash
# docker rm rayt

FROM raytraverse

WORKDIR /working

SHELL ["/bin/bash", "-c"]
COPY raytraverse-*-py3-none-any.whl ./
RUN pip3 install *.whl
RUN rm *.whl

CMD raytraverse --help