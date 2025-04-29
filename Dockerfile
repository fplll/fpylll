## -*- docker-image-name: "fplll/fpylll" -*-

FROM fplll/fplll
MAINTAINER Martin Albrecht <fplll-devel@googlegroups.com>

ARG BRANCH=master
ARG JOBS=2
ARG CXXFLAGS="-O2 -march=x86-64"
ARG CFLAGS="-O2 -march=x86-64"

SHELL ["/bin/bash", "-c"]
ENTRYPOINT /usr/local/bin/ipython

RUN apt update && \
    apt install -y python3-pip python3-dev zlib1g-dev libjpeg-dev && \
    apt clean

RUN rm /usr/lib/python3.11/EXTERNALLY-MANAGED

RUN git clone --branch $BRANCH https://github.com/fplll/fpylll

RUN pip3 install -r fpylll/requirements.txt
RUN pip3 install -r fpylll/suggestions.txt

RUN cd fpylll && \
    CFLAGS=$CFLAGS CXXFLAGS=$CXXFLAGS python3 setup.py build -j $JOBS && \
    pip3 install . && \
    cd .. && \
    rm -rf fpylll

