# syntax=docker/dockerfile:1

FROM raytmany:latest

SHELL ["/bin/bash", "-c"]

WORKDIR /app
ADD ./dist/raytraverse-1.3.0.tar.gz ./
WORKDIR /app/raytraverse-1.3.0

CMD ["/bin/bash", "build_scripts/docker_run.sh", "cp37-cp37m", "cp38-cp38" "cp39-cp39" "cp310-cp310"]