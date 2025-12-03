FROM tensorflow/tensorflow:2.16.2-gpu

ARG UID=1000
ARG GID=1000
ARG USER=user
ARG GROUP=user

# Set environment variable to avoid interactive timezone configuration
ENV TZ=Etc/UTC
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Install essential build tools and dependencies
RUN set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    pkg-config \
    sudo \
    rsync; \
    apt-get clean; \
    rm -rf /var/lib/apt/lists/*

# Install Python 3.10 from source
# 1) Tools & build deps
RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential wget ca-certificates curl \
      libssl-dev zlib1g-dev libbz2-dev libreadline-dev libsqlite3-dev \
      libffi-dev liblzma-dev libncursesw5-dev libgdbm-dev tk-dev uuid-dev \
 && rm -rf /var/lib/apt/lists/*

# 2) Build and install Python 3.10.x
ARG PY_VER=3.10.18
RUN set -eux; \
    wget -O /tmp/Python-${PY_VER}.tar.xz https://www.python.org/ftp/python/${PY_VER}/Python-${PY_VER}.tar.xz; \
    cd /tmp && tar -xf Python-${PY_VER}.tar.xz; \
    cd /tmp/Python-${PY_VER} && ./configure --enable-optimizations --with-ensurepip=install; \
    make -j"$(nproc)" && make install; \
    rm -rf /tmp/Python-${PY_VER} /tmp/Python-${PY_VER}.tar.xz

RUN update-alternatives \
    --install /usr/bin/python python /usr/local/bin/python3 1

# Verify Python installation and venv module
RUN python --version && python -m venv --help

# Install pip for Python 3.10 using get-pip.py
RUN set -eux; \
    curl -sS https://bootstrap.pypa.io/get-pip.py -o /tmp/get-pip.py; \
    echo "dffc3658baada4ef383f31c3c672d4e5e306a6e376cee8bee5dbdf1385525104 /tmp/get-pip.py" | sha256sum -c -; \
    python3.10 /tmp/get-pip.py; \
    rm /tmp/get-pip.py

# Create group + user
RUN set -eux; \
    groupadd --gid "${GID}" "${GROUP}"; \
    useradd --uid "${UID}" \
    --gid "${GID}" \
    --shell /bin/bash \
    --create-home \
    "${USER}" && \
    usermod -aG sudo "${USER}" && \
    echo "${USER} ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/"${USER}" && \
    chmod 0440 /etc/sudoers.d/"${USER}"

# Create the build cache
RUN mkdir -p /build_cache && chown -R ${USER}:${GROUP} /build_cache
VOLUME /build_cache

USER ${USER}:${GROUP}

# Set working directory
WORKDIR /workspace

COPY --chown=${USER}:${GROUP} . .