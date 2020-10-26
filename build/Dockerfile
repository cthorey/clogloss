ARG CUDAGL_TAG
FROM nvidia/cudagl:$CUDAGL_TAG

# supported NVIDIA Architecture
# Should be smth torch_cuda_arch_list="6.1;7.5" - where the architecture can be
# found using sudo nvidia-container-cli --load-kmods info
ARG TORCH_CUDA_ARCH_LIST

# change shell
RUN rm /bin/sh && ln -s /bin/bash /bin/sh

###############################
## INSTALL / SET UP DEPENDENCIES
###############################
COPY build/requirements /build

# apt install
RUN apt update && \
  cat /build/apt | xargs apt install --allow-unauthenticated --yes && \
  apt clean

# pip install
RUN python3.6 -m pip install --requirement /build/pip3
# configure ipython
COPY build/ipython_config.py /root/.ipython/profile_default/ipython_config.py
# configure jupyter
COPY build/jupyter_notebook_config.py /root/.jupyter/

###############################
## COPY SOURCECODE
###############################

COPY ./vegai /packages/vegai
COPY ./scripts /vegai/scripts
COPY ./build/packages/orm /packages/orm
###############################
## SETUP ENVIRONMENT
###############################
ENV ROOT_DIR=/workdir
ENV PASSWORD=tf

ENV QT_X11_NO_MITSHM=1
ENV PYTHONPATH=$PYTHONPATH:/packages
ENV PATH=$PATH:/workdir/scripts
ENV TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST

WORKDIR /workdir

CMD ["bash"]
