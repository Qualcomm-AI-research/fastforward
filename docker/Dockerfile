ARG VER_CUDA=13.0
FROM nvidia/cuda:${VER_CUDA}.0-base-ubuntu22.04

ENV VIRTUAL_ENV=/opt/venv \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_PYTHON_INSTALL_DIR=/opt/uv-python \
    REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt

RUN apt update && \
    DEBIAN_FRONTEND=noninteractive apt install --yes --no-install-recommends \
        acl \
        bash-completion \
        ca-certificates \
        curl \
        fzf \
        git \
        jq \
        less \
        nodejs \
        npm \
        openssh-client \
        openssh-server \
        ripgrep \
        shellcheck \
        sudo \
        vim \
    && rm -rf /var/lib/apt/lists/*

RUN curl -L 'https://pki.qualcomm.com/{qc_root_g2_cert.crt,ssl_v4_cert.crt}' -o '/usr/local/share/ca-certificates/#1' \
    && update-ca-certificates

RUN curl -L "https://github.com/astral-sh/uv/releases/latest/download/uv-x86_64-unknown-linux-gnu.tar.gz" \
        | tar --preserve-permissions --directory /usr/local/bin --strip-components=1 -x -z -f -

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

RUN curl -L "https://github.com/neovim/neovim/releases/download/v0.10.4/nvim-linux-x86_64.tar.gz" | tar --preserve-permissions --directory /usr/local/ --strip-components=1 -x -z -f - \
    && curl -L "https://github.com/jesseduffield/lazygit/releases/download/v0.47.2/lazygit_0.47.2_Linux_x86_64.tar.gz" | tar --preserve-permissions --directory /usr/local/bin -x -z -f - lazygit

ARG VER_PYTHON \
    VER_TORCH \
    VER_CUDA

RUN uv python install ${VER_PYTHON} \
    && uv venv --python ${VER_PYTHON} ${VIRTUAL_ENV} \
    && chmod -R 777 ${VIRTUAL_ENV} ${UV_PYTHON_INSTALL_DIR} \
    && setfacl -R -d -m o::rwx ${VIRTUAL_ENV}

ENV PATH=${VIRTUAL_ENV}/bin:$PATH

RUN export VER_CUDA_SHORT="$(echo $VER_CUDA | awk -F'.' '{print $1$2}')" \
    && uv pip install --no-cache torch==${VER_TORCH}+cu${VER_CUDA_SHORT} \
        --extra-index-url="https://download.pytorch.org/whl/cu${VER_CUDA_SHORT}"

COPY pyproject.toml /workspace/

# --inexact keeps the torch installed above out of the sync.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --no-install-project --inexact \
        --project /workspace \
        --no-default-groups --group dev --group test --group docs

ENV PYTHONPYCACHEPREFIX=/tmp
ENTRYPOINT (set -x ; ls ~/.vscode-server/ || sudo mkdir -m a=rwx -p "$(readlink ~/.vscode-server)" ; ls ~/.config/JetBrains/ || sudo mkdir -m a=rwx -p "$(readlink ~/.config/JetBrains)") ; ./scripts/rebuild && exec $0 "$@"
CMD bash
