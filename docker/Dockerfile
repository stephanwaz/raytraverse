# syntax=docker/dockerfile:1

# docker build . --tag raytraverse:latest --tag raytraverse:v1.3.5
# docker run -it --name rayt --mount type=bind,source="$(pwd)",target=/working raytraverse /bin/bash
# docker rm rayt

FROM raytraverse

WORKDIR /build

SHELL ["/bin/bash", "-c"]
RUN apt-get update
RUN apt-get -y install man

COPY raytraverse-*-py3-none-any.whl ./
RUN pip3 install --force-reinstall --no-deps *.whl
RUN rm *.whl

RUN curl -s https://api.github.com/repos/LBNL-ETA/Radiance/releases\?per_page\=1 \
    | grep "browser_download_url.*Linux.zip" | cut -d: -f2,3 | tr -d \" | wget -i -
RUN unzip Radiance_*_Linux.zip
RUN tar -xzf radiance-*-Linux.tar.gz
WORKDIR /radiance
RUN rm -rf bin lib man
RUN mv /build/radiance-*-Linux/usr/local/radiance/* ./
RUN rm -rf /build

ENV RAYPATH=.:/radiance/lib
ENV MANPATH=/radiance/man
ENV PATH=/radiance/bin:$PATH
RUN raytraverse --help
WORKDIR /working
