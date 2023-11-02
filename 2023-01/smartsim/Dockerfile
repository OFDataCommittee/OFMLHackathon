# Dockerfile to build SmartSim/SmartRedis-ready images for developing OpenFOAM-related stuff
#FROM opencfd/openfoam-dev
# The images from OpenCFD do not play nice with Github, so I use my own ones
FROM ghcr.io/foamscience/jammy-openfoam:v2212
LABEL maintainer="Mohammed Elwardi Fadeli <elwardi.fadeli@tu-darmstadt.de>"

# Dependencies
USER root
RUN apt update && apt install -y python3-dev python3-pip git-core git-lfs cmake unzip vim wget
RUN apt clean && apt autoremove --yes && rm -rf /var/lib/{cache,log}/

USER openfoam

# Software versions; overriding these has no effect on resulting containers; but nice to have
# as environment variables to remove version confusion
ENV SMARTREDIS_VERSION="6bfcf1deb0d726df75610fe8a352b9ecb13c44f7"
ENV SMARTSIM_VERSION="0.5.1"

# Install SmartSim (maybe get the development version for this too?)
RUN pip3 install -U smartsim[ml]==$SMARTSIM_VERSION
ENV PATH=$PATH:/home/openfoam/.local/bin

# Build SmartRedis
ENV FOAM_SMARTREDIS=$HOME/smartredis
RUN git clone https://github.com/CrayLabs/SmartRedis $FOAM_SMARTREDIS &&\
    cd $FOAM_SMARTREDIS && git reset --hard $SMARTREDIS_VERSION
RUN cd $FOAM_SMARTREDIS && smart clobber && smart build --device=cpu && make lib

# Environment variables to use for linking SmartRedis libs to OpenFOAM bins/libs
ENV SMARTREDIS_INCLUDE=$FOAM_SMARTREDIS/install/include
ENV SMARTREDIS_LIB=$FOAM_SMARTREDIS/install/lib
ENV LD_LIBRARY_PATH=$SMARTREDIS_LIB:$LD_LIBRARY_PATH
RUN find /usr/lib/openfoam -iname DiagonalMatrix.H -exec sudo sed -i 's/DiagonalMatrix<Type>/DiagonalMatrix/g' {} \;
RUN sudo chmod 777 -R /home/openfoam/data
