## -*- docker-image-name: "fplll/fpylll" -*-

FROM fplll/fplll
MAINTAINER Martin Albrecht <fplll-devel@googlegroups.com>

ARG BRANCH=master

RUN apt update && \
    apt install -y python3-pip python3-dev && \
    apt clean && \
    git clone --branch $BRANCH https://github.com/fplll/fpylll && \
    cd fpylll && \
    pip3 install Cython && \
    pip3 install -r requirements.txt && \
    pip3 install -r suggestions.txt && \
    python3 setup.py build && \
    python3 setup.py -q install 
