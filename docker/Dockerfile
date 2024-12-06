FROM ubuntu:22.04

ENV CONDA_DEFAULT_ENV=dev \
    CONDA_PREFIX=/opt/conda \
    REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt

RUN apt update && \
    DEBIAN_FRONTEND=noninteractive apt install --yes --no-install-recommends \
        acl \
        ca-certificates \
        curl \
        fzf \
        git \
        jq \
        nodejs \
        npm \
        openssh-client \
        openssh-server \
        ripgrep \
        shellcheck \
        sudo \
        vim \
    && rm -rf /var/lib/apt/lists/* \
    && curl -L 'https://pki.qualcomm.com/{qc_root_g2_cert.crt,ssl_v3_cert.crt,ssl_v4_cert.crt}' -o '/usr/local/share/ca-certificates/#1' \
    && update-ca-certificates \
    && curl -o /tmp/conda.sh -L 'https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh' \
    && mkdir -m 777 -p ${CONDA_PREFIX} \
    && setfacl -d -m o::rwx ${CONDA_PREFIX} \
    && bash /tmp/conda.sh -u -b -p ${CONDA_PREFIX} \
    && rm /tmp/conda.sh \
    && ${CONDA_PREFIX}/bin/conda config --set channel_priority strict \
    && ${CONDA_PREFIX}/bin/conda init --no-user --system --all \
    && echo "conda activate $CONDA_DEFAULT_ENV" >> /etc/profile.d/conda.sh

RUN echo '%users ALL = (ALL) NOPASSWD: ALL' > /etc/sudoers.d/passwordless \
    && echo '\
if [ -f $HOME/.profile.local ] ; then\n\
  . $HOME/.profile.local\n\
fi\n\
' > /usr/local/etc/profile.global \
    && echo '\
env | grep -v -P "^(_|HOSTNAME|PS1|PWD|SHELL|SHLVL|TERM|USER)=" | sudo tee -a /etc/environment > /dev/null\n\
sudo mkdir -p /run/sshd\n\
sudo /usr/sbin/sshd "$@"\n\
' > /usr/local/bin/run_sshd

RUN curl -L "https://github.com/neovim/neovim/releases/download/v0.10.0/nvim-linux64.tar.gz" | tar --preserve-permissions --directory /usr/local/ --strip-components=1 -x -z -f - \
    && curl -L "https://github.com/jesseduffield/lazygit/releases/download/v0.43.1/lazygit_0.43.1_Linux_x86_64.tar.gz" | tar --preserve-permissions --directory /usr/local/bin -x -z -f - lazygit

ARG VER_PYTHON=3.10 \
    VER_TORCH=2.3.0 \
    VER_CUDA=11.8.0

RUN ${CONDA_PREFIX}/bin/conda create --quiet --name ${CONDA_DEFAULT_ENV} -y --strict-channel-priority \
        python=${VER_PYTHON} \
        python-build \
        cuda-gdb \
        cuda-nvcc \
        cuda-nvtx \
        cuda-libraries-dev \
        -c nvidia/label/cuda-${VER_CUDA} \
    # Install torch via pip to not depend on mkl
    && export VER_CUDA_SHORT="$(echo $VER_CUDA | awk -F'.' '{print $1$2}')" \
    && ${CONDA_PREFIX}/bin/conda run  --name ${CONDA_DEFAULT_ENV} --live-stream \
        python3 -m pip install --no-cache-dir torch==${VER_TORCH}+cu${VER_CUDA_SHORT} --extra-index-url="https://download.pytorch.org/whl/cu${VER_CUDA_SHORT}" \
    && ${CONDA_PREFIX}/bin/conda clean --yes --all --verbose \
    && rm -rf ~/.conda*

COPY pyproject.toml /workspace/

RUN export PIP_CACHE_DIR=/tmp/pip-cache \
    && export BUILD_DIR=/tmp/build \
    && mkdir -p $BUILD_DIR \
    && ${CONDA_PREFIX}/bin/conda run --name ${CONDA_DEFAULT_ENV} --live-stream bash -c "set -x -o pipefail; \
        python3 -m pip install --dry-run --ignore-installed  --report $BUILD_DIR/pip-report.json -v -C build-dir=$BUILD_DIR/{wheel_tag} /workspace[dev,test,docs] ; \
        jq '.install[0].metadata.requires_dist[] | split(\";\") | .[0]' $BUILD_DIR/pip-report.json | xargs -i python3 -m pip install {} \
    " \
    && rm -rf $BUILD_DIR \
    && rm -rf $PIP_CACHE_DIR

ENV PYTHONPYCACHEPREFIX=/tmp
ENTRYPOINT (set -x ; ls ~/.vscode-server/ || sudo mkdir -m a=rwx -p "$(readlink ~/.vscode-server)" ; ls ~/.config/JetBrains/ || sudo mkdir -m a=rwx -p "$(readlink ~/.config/JetBrains)") ; . /etc/profile.d/conda.sh ; ./scripts/rebuild && exec $0 "$@"
CMD bash
