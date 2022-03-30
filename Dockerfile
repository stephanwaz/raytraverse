# syntax=docker/dockerfile:1

FROM raytdev:current

SHELL ["/bin/bash", "-c"]

WORKDIR /app
ADD ./dist/raytraverse-1.2.8.tar.gz ./
WORKDIR /app/raytraverse-1.2.8

RUN python3 setup.py bdist_wheel
RUN pip3 install dist/raytraverse-*.whl

CMD ["/bin/bash", "docker_run.sh"]