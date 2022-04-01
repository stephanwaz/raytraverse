# syntax=docker/dockerfile:1


# docker build --tag raytmany .
FROM quay.io/pypa/manylinux_2_24_x86_64

WORKDIR /requirements

COPY requirements.txt requirements.txt

RUN /opt/python/cp37-cp37m/bin/pip install -r requirements.txt
RUN /opt/python/cp38-cp38/bin/pip install -r requirements.txt
RUN /opt/python/cp39-cp39/bin/pip install -r requirements.txt
RUN /opt/python/cp310-cp310/bin/pip install -r requirements.txt