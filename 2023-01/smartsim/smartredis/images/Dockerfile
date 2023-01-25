#
# BSD 2-Clause License
#
# Copyright (c) 2021-2023, Hewlett Packard Enterprise
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#

## Builder image
FROM ubuntu:20.04 as builder

# Install tooling
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -q -y --no-install-recommends \
    ca-certificates git \
    make cmake gcc g++ libc6-dev python3 libpython3-dev pip gfortran && \
    rm -rf /var/lib/apt/lists/*
RUN ln -s /usr/bin/python3 /usr/bin/python

# Get source code from current clone
COPY . /usr/local/src/SmartRedis

# To build a docker image from another repository,
# users should uncomment the following WORKDIR and
# RUN commands and remove the COPY command above.
#WORKDIR /usr/local/src/
#RUN git config --global advice.detachedHead false && \
#    git clone https://github.com/CrayLabs/SmartRedis.git --branch develop

# Compile and install
WORKDIR /usr/local/src/SmartRedis
RUN make clobber && make lib && pip install . && rm -rf ~/.cache/pip \
    && rm -rf build tests examples images utils third-party doc images

# Copy install files and directories to /usr/local/lib and
# usr/local/include and delete unnecessary build files
RUN cp -r /usr/local/src/SmartRedis/install/lib/* /usr/local/lib/ && \
    mkdir /usr/local/include/smartredis && \
    cp -r /usr/local/src/SmartRedis/install/include/* /usr/local/include/smartredis/
