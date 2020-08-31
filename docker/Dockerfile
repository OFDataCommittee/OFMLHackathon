FROM continuumio/anaconda3:5.0.1

# install sys requirements
RUN apt-get update
RUN apt-get install -y build-essential git

# clone in repo
RUN git clone https://github.com/spartee/silc

# update conda
RUN conda update --yes conda

# go into repo
RUN cd silc && conda env create -f silc-dev-requirements.yml

# activate
RUN echo "source activate silc-dev-requirements" > ~/.bash
ENV PATH /opt/conda/envs/silc-dev-requirements/bin:$PATH

# activate
#RUN conda activate silc-dev-requirements

# make build dir
RUN cd silc && mkdir build

# run cmake
RUN  cd silc/build && \
    cmake .. -DCMAKE_INSTALL_PREFIX=$(conda info --base)/envs/silc-dev-requirements

# build and install project
RUN cd silc/build && \
    make -j2 install