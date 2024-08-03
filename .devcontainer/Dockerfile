FROM nvidia/cuda:11.6.2-base-ubuntu20.04

ENV PYTHON_VERSION=3.10

ENV PATH /opt/conda/bin:$PATH
ENV LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/usr/local/lib"
ENV LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/opt/conda/lib"

ENV PYTHONIOENCODING=UTF-8
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive
ENV CONDA_AUTO_UPDATE_CONDA=false

ARG UID
ARG GID

ENV UID=${UID:-1000}
ENV GID=${GID:-1000}

# install basic tools
RUN apt update
RUN apt install -y bash \
    build-essential \
    git \
    curl \
    ca-certificates \
    wget \
    sudo \
    && rm -rf /var/lib/apt/lists

# Install Miniconda and create main env
ADD https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh miniconda3.sh
RUN /bin/bash miniconda3.sh -b -p /opt/conda \
    && rm miniconda3.sh \
    && /opt/conda/bin/conda install -y -c anaconda python=$PYTHON_VERSION \
    && /opt/conda/bin/conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.6 -c pytorch -c nvidia \
    && /opt/conda/bin/conda clean -ya

RUN /opt/conda/bin/conda config --set ssl_verify False \
    && pip install --upgrade pip --trusted-host pypi.org --trusted-host files.pythonhosted.org \
    && ln -s /opt/conda/bin/pip /usr/local/bin/pip3

# change uid and gid of user jupyter
# create user jupyter (uid 1000, gid 1000)
RUN addgroup --system --gid ${GID} jupyter && \
    adduser jupyter --system --uid ${UID} --gid ${GID} --shell /usr/bin/fish

# add user to sudoers
RUN echo "jupyter ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

# clean up
RUN apt-get clean && \
    rm -rf /var/lib/apt/lists/*

USER jupyter

RUN pip install --upgrade pip

# install lint tools
RUN pip install flake8 \
    black \
    isort

# install required packages
RUN pip install matplotlib scikit-learn opencv-python-headless pyyaml libtiff

CMD ["/bin/bash"]
